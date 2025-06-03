# [Overnight Map](https://www.247map.app)

[This](https://finviz.com/map.ashx), but with 24/5 data where applicable. Hosted on a Google Cloud VM (Compute Engine) instance.

To do list:
- add basic db to track # of unique visits
- fix scaling so figures can actually render on mobile
- add auto-start on boot / auto-restart on crash (via systemd; currently using tmux + manual reboots if something goes wrong)
- optimize w/ local memory caching by serving pre-rendered Static HTML (pre-generating the Plotly heatmap as a standalone HTML file every 5 minutes; then serve that file statically using a web server or object store, probably GCS)
- add logging for GCP (so can view logs without having to SSH in)

Some samples of what this looks like (these may be outdated, but the general idea is there).

![image](https://github.com/user-attachments/assets/29f388d5-c883-4322-8f7a-cf39875b97ff)


![image](https://github.com/user-attachments/assets/a061112e-0a63-419a-93c5-5c64ee9fd3c1)
