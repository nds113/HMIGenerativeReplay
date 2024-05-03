import os
import faiss
import torch
import numpy as np

from torch.utils.data import DataLoader

class HippocampalMemory(object):
    def __init__(self, feature_dim, max_size=None, vector_split=16, num_codebooks=256):

        self.memories = {}
        self.feature_dim = feature_dim
        self.max_size = max_size
        codebook_bits = int(np.log2(num_codebooks))
        self.produce_quantizer = faiss.ProductQuantizer(feature_dim, vector_split, codebook_bits)

    def set_new_task(self, task):
        """Instantiate a new task in the memory index"""
        self.current_task = task
        self.memories[task] = torch.empty(0, self.feature_dim)

    def convert_to_code(self, features):
        return self.produce_quantizer.compute_codes(features.numpy().astype('float32'))

    def reconstruct_feature(self, code):
        return torch.tensor(self.produce_quantizer.decode(code), dtype=torch.float32)
    
    def reconstruct_feature_from_memory(self, feature):
        feature_shape = feature.shape
        reconst = self.reconstruct_feature(self.convert_to_code(feature.reshape(-1, self.feature_dim)))
        return reconst.reshape(feature_shape)

    def memorize(self, new_features):
        """Concatenate new task features to the memory index of associated with this task"""
        new_features = new_features.detach().cpu().reshape(-1, 768)
        self.memories[self.current_task] = torch.cat((self.memories[self.current_task], new_features))

    def prune_memory(self):
        """Prune memories if we are limiting the total size of the memory index"""
        if self.max_size is not None:
            # Max size should be equal across all tasks
            max_task_memory_size = self.max_size // len(self.memories)

            # Loop through tasks prune memories that exceed maximum size
            for task in self.memories.keys():
                if self.memories[task] > max_task_memory_size:
                    # Select 
                    keep_indices = self.select_retrieval_indices(task, max_task_memory_size)
                    self.memories[task] = self.memories[task][keep_indices]
    
    def retrieve(self, task, num_memories):
        """Retrieve a number of memories associated with given task -
        stored memories are codes that are then reconstructed 
        """
        retrieval_indices = self.select_retrieval_indices(task, num_memories)
        retrieved_memories = self.memories[task][retrieval_indices]
        return self.reconstruct_feature(retrieved_memories)
    
    def select_retrieval_indices(self, task, num_memories):

        """Retrieve memories based on K-Means clustering"""
        if len(self.memories[task]) < num_memories:
            return torch.randint(len(self.memories[task]), (num_memories,))

        # Reconstruct all memories associated with desired task 
        reconstructed_memories = (self.reconstruct_feature(self.memories[task]).numpy().astype("float32"))

        # Instantiate and train a K-Means clusterer based on the number of memories we're retrieving
        kmeans_clusterer = faiss.Kmeans(d=self.feature_dim, k=num_memories)
        kmeans_clusterer.train(reconstructed_memories)

        # Retrieve nearest neighbor memories
        _, nearest_neighbor_indices = faiss.knn(kmeans_clusterer.centroids, reconstructed_memories, 1)
        return nearest_neighbor_indices.squeeze()

    def quantize_memory(self):
        """Quantize a new memory - reconstruct all coded memories, retrain codebooks, then quantize memories again."""

        for task, memory_index in self.memories.items():
            # Reconstruct coded memories
            if not isinstance(memory_index, torch.Tensor):
                self.memories[task] = self.reconstruct_feature(memory_index)

        # Extract all memories and retrain codebook quantizer
        self.produce_quantizer.train(torch.cat(list(self.memories.values())).numpy().astype('float32'))

        # Re-quantize all memories
        self.memories = {task: self.convert_to_code(feature) for task, feature in self.memories.items()}

    def save_pretrained(self, codebook_dir):

        codebook = faiss.vector_to_array(self.produce_quantizer.centroids).astype("float32")
        np.save(os.path.join(codebook_dir, "codebook.npy"), codebook)
        torch.save(self.memories, os.path.join(codebook_dir, "memories"))

    def load_pretrained(self, codebook_dir):

        codebook = np.load(os.path.join(codebook_dir, "codebook.npy"))
        faiss.copy_array_to_vector(codebook.astype("float32"), self.produce_quantizer.centroids)
        self.memories = torch.load(os.path.join(codebook_dir, "memories"))
    
    @property
    def memory_size(self):
        """Return memory lengths"""
        return {task: len(memory) for task, memory in self.memories.items()}

    @property
    def memory_byte_size(self):
        """Return byte size of memories"""

        memory_byte_size = {}
        for task, memory_index in self.memories.items():
            # Calculate byte size of PyTorch tensor entries
            if isinstance(memory_index, torch.Tensor):
                memory_byte_size[task] = (memory_index.element_size()*memory_index.nelement())
            # Retrieve byte size of Numpy array entries
            elif isinstance(memory_index, np.ndarray):
                memory_byte_size[task] = memory_index.nbytes
            else:
                raise AssertionError
            
        # Calculate codebook size
        memory_byte_size["codebook"] = faiss.vector_to_array(self.produce_quantizer.centroids).astype("float32").nbytes
        return memory_byte_size
    
def pad_to_max_len(seq, pad_len, val):
    return seq + [val] * pad_len


def pad_all_to_max_len(batch, val):
    max_len = max(len(seq) for seq in batch)
    return [pad_to_max_len(seq, max_len - len(seq), val) for seq in batch]

def varlen_collate_fn(data, decoder_special_token_ids=None, encoder_special_token_ids=None, n_gpus=1):

    batch_size = (len(data) + n_gpus - 1) // n_gpus
    cqs = torch.tensor(
        pad_all_to_max_len(
            [datum["dec_examples"]["cq"] for datum in data],
            decoder_special_token_ids["pad_token"],
        )
    )
    len_cqs = torch.tensor([datum["dec_examples"]["len_cq"] for datum in data])
    cqas = torch.tensor(
        pad_all_to_max_len(
            [datum["dec_examples"]["cqa"] for datum in data],
            decoder_special_token_ids["pad_token"],
        )
    )
    len_cqas = torch.tensor([datum["dec_examples"]["len_cqa"] for datum in data])
    qa_Ys = torch.tensor(
        pad_all_to_max_len([datum["dec_examples"]["qa_Y"] for datum in data], -1)
    )
    gen_Ys = torch.tensor(
        pad_all_to_max_len([datum["dec_examples"]["gen_Y"] for datum in data], -1)
    )
    is_replays = torch.tensor([datum["is_replay"] for datum in data])

    if data[0]["enc_examples"] is not None:
        cls_cq = torch.tensor(
            pad_all_to_max_len(
                [datum["enc_examples"]["cls_cq"] for datum in data],
                encoder_special_token_ids["pad_token"],
            )
        )
        return (
            cls_cq,
            cqs,
            len_cqs,
            cqas,
            len_cqas,
            qa_Ys,
            gen_Ys,
            is_replays,
        )

    return (
        None,
        cqs,
        len_cqs,
        cqas,
        len_cqas,
        qa_Ys,
        gen_Ys,
        is_replays,
    )


def memorize_task_features(data, encoder, memory_module, decoder_special_token_ids, encoder_special_token_ids, memory_fraction=0.25):

    def collate_fn(x, encoder_special_token_ids=encoder_special_token_ids, decoder_special_token_ids=decoder_special_token_ids):
        return varlen_collate_fn(x, decoder_special_token_ids=decoder_special_token_ids, encoder_special_token_ids=encoder_special_token_ids)

    dataloader = DataLoader(data, collate_fn=collate_fn, shuffle=True, batch_size=48)
    with torch.no_grad():
        for (cls_cqs,_,_,cqas,len_cqas,_,_,is_replayed) in dataloader:
            cls_cqs.cuda()
            encoded = encoder(cls_cqs.cuda())
            encoded_mem = encoded[0].cpu()

            # Compute the number of samples to memorize based on the memory_fraction
            num_samples_to_memorize = int(memory_fraction * len(encoded_mem))

            # Select a random subset of samples to memorize
            indices_to_memorize = torch.randperm(len(encoded_mem))[:num_samples_to_memorize]

            # Memorize only the selected subset of samples
            memory_module.memorize(encoded_mem[indices_to_memorize].detach())

            #memory_module.memorize(encoded_mem[is_replayed.logical_not()].detach())





    
