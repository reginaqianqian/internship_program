from fake_hw import run_fake_hw

import time
import pytest
import json

import numpy as np

user_got_frames = 0
def test_hstouch():
    global user_got_frames
    from tactile_sdk.sharpa.tactile import Touch
    import multiprocessing

    CHANNEL = range(10)
    touch = Touch('127.0.0.1', 50001, CHANNEL, [])

    # test start() and stop()
    for _ in range(10):
        assert touch.start()
        assert touch.stop()

    # test obtain data with fetch()
    user_got_frames = 0
    touch.start()
    process_fake_hw = multiprocessing.Process(target=run_fake_hw)
    process_fake_hw.start()
    for _ in range(20):
        for ch in CHANNEL:
            time.sleep(0.01)
            ret = touch.fetch(ch, timeout=0.01)
            if ret is None: continue
            # _, img, _ = ret
            img=ret["content"]["RAW"]
            # assert img.shape == (240, 320, 1)
            user_got_frames += 1
    process_fake_hw.join()
    assert user_got_frames > 0  # consider potential frame loss
    touch.stop()
    # test obtain data with callback
    user_got_frames = 0

    def callback(ch, ts, img, inferred):
        global user_got_frames
        # assert img.shape == (240, 320, 1)
        user_got_frames += 1

    touch = Touch('127.0.0.1', 50001, CHANNEL, [], callback=callback)
    touch.start()
    process_fake_hw = multiprocessing.Process(target=run_fake_hw)
    process_fake_hw.start()
    process_fake_hw.join()
    touch.stop()
    assert user_got_frames > 10
