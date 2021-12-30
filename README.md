# [Fashion Killer](https://www.youtube.com/watch?v=nPhe9Ah8rKA)
![](FK.gif)  
^^^ The above is dynamic and can do appearance transfer on any input images, not just the seven examples shown.

Quite a lot going on here:
- **Scrape from asos.com:** prepare_asos_data/
- **Use PyTorch pretrained pose detector:** pose_drawer/
- **Train [SA-GAN](https://arxiv.org/abs/1805.08318) using custom attention mechanism:** app_transfer/
- **Putting them both together to perform appearance transfer:** rtc_server/monkey.py
- **Front-end web app:** rtc_server/client.js
- **RTC server:** rtc_server/server.py
 
I'm particularly proud of the RTC server. It streams data from the webcam straight into a remote GPU server. It then uses custom threading, queueing and activation caching so the app can run at a decent frame rate in real-time (see gif above). 
