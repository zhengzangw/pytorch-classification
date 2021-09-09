import numpy as np
from faiss import Kmeans as faiss_Kmeans
from tqdm import tqdm

from ..utils import utils

log = utils.get_logger(__name__)


class Kmeans(object):
    DEFAULT_KMEANS_SEED = 1234

    def __init__(
        self,
        k_list,
        data,
        epoch=0,
        init_centroids=None,
        frozen_centroids=False,
        name="",
    ):
        """
        Performs many k-means clustering.

        Args:
            k_list (List[int]): list of k
            data (np.array): data [N x dim] to cluster
            epoch: random seed
            init_centroids: initial centroids to use
        """
        super().__init__()
        self.k_list = k_list
        self.data = data
        self.d = data.shape[-1]
        self.init_centroids = init_centroids
        self.frozen_centroids = frozen_centroids
        self.name = name

        self.debug = False
        self.epoch = epoch + 1

    def compute(self, ret_label=False):
        """compute cluster

        Returns:
            dict
        """
        data = self.data
        labels = []
        centroids = []

        tqdm_batch = tqdm(total=len(self.k_list), desc=f"[K-means {self.name}]", leave=False)
        for k_idx, each_k in enumerate(self.k_list):
            seed = k_idx * self.epoch + self.DEFAULT_KMEANS_SEED
            kmeans = faiss_Kmeans(
                self.d,
                each_k,
                niter=40,
                verbose=False,
                spherical=True,
                min_points_per_centroid=1,
                max_points_per_centroid=10000,
                gpu=False,
                seed=seed,
                frozen_centroids=self.frozen_centroids,
            )

            kmeans.train(data, init_centroids=self.init_centroids)

            if ret_label:
                _, clus_label = kmeans.index.search(data, 1)
                labels.append(clus_label.squeeze(1))
            C = kmeans.centroids
            centroids.append(C)

            tqdm_batch.update()
        tqdm_batch.close()

        if ret_label:
            labels = np.stack(labels, axis=0)
        ret = dict(centroid=centroids, label=labels)
        return ret


class MemQueue:
    # cityscapes: 2975 x 22000
    # gta5: 24967 x 21749
    # 2679 sp for one image

    def __init__(self, size=2500 * 500, dim=256, name=""):
        super().__init__()

        self.size = size
        self.dim = dim
        self.tail = 0
        self.filled = False
        self.ready = False
        self.name = name

        self.memory_queue = np.zeros((size, dim), dtype=np.float32)

    def _push(self, features):
        length = len(features)
        assert length < self.size
        left = self.tail
        right = (left + length) % self.size
        if left < right:
            self.memory_queue[left:right] = features
        else:
            # log.info(f"[Memqueue] Filled with tail={self.tail}, input length={length}")
            self.filled = True
            self.ready = True
            mid = self.size - left
            self.memory_queue[left:] = features[:mid]
            self.memory_queue[:right] = features[mid:]

        self.tail = right

    def push(self, features, sample_ratio=0.5):
        # features [N x dim]
        if sample_ratio < 1:
            size = len(features)
            sample_num = int(size * sample_ratio)
            mask = np.random.choice(size, sample_num, replace=False)
            features = features[mask]
        self._push(features)

    def protos(self, k_list, **kwargs):
        km = Kmeans(k_list, self.memory_queue, name=self.name, **kwargs)
        ret = km.compute()
        # after compute the cluster, waiting for new features
        self.ready = False
        return ret["centroid"]

    def __len__(self):
        return len(self.memory_queue)

    @property
    def size_gb(self):
        return self.memory_queue.nbytes // (1024 * 1024 * 1024)

    def __repr__(self) -> str:
        return self.memory_queue.__repr__()

    def __str__(self) -> str:
        return self.memory_queue.__str__()

    def randomize(self):
        self.memory_queue = np.random.rand(self.size, self.dim).astype("float32")
        self.filled = True
        return self
