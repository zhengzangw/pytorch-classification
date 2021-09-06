import numpy as np

from src.modules.memqueue import MemQueue


def test_mem_size():
    m = MemQueue()


def test_loop_push():
    m = MemQueue(size=10, dim=1)
    fake = np.arange(10)[:, None]
    m.push(fake[:6])
    assert not m.filled
    m.push(fake[:6])
    assert m.memory_queue[0] == 4
    m.push(fake[:9])
    assert m.memory_queue[0] == 8


def test_clus():
    m = MemQueue().randomize()
    proto = m.protos([19] * 5 + [38] * 5)
