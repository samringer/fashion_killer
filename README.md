# [Fashion Killer](https://www.youtube.com/watch?v=nPhe9Ah8rKA)
![](FK.gif)
^^^ The above is dynamic and can do apperance transfer on any input images, not just the seven examples shown.

Quite a lot going on here:
- **Scrape from asos.com:** prepare_asos_data/
- **Use PyTorch pretrained pose detector:** pose_drawer/
- **Train [SA-GAN](https://arxiv.org/abs/1805.08318) using custom attention mechanism:** app_transfer/
- **Putting them both together to perform appearance transfer:** rtc_server/monkey.py
- **Front-end to app:** rtc_server/client.js
- **RTC server:** rtc_server/server.py
