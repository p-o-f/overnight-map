#!/bin/bash

# Exit on error
set -e

# Install dependencies
sudo apt update
sudo apt install -y python3 python3-pip git \
    chromium-browser chromium-chromedriver build-essential unzip wget

# Optional: symlinks for compatibility
sudo ln -sf /usr/bin/chromedriver /usr/local/bin/chromedriver
sudo ln -sf /usr/bin/chromium-browser /usr/bin/google-chrome

# Clone your repo
cd ~
if [ -d overnight-map ]; then
  echo "Directory overnight-map already exists, pulling latest..."
  cd overnight-map && git pull
else
  git clone https://github.com/p-o-f/overnight-map.git
  cd overnight-map
fi

# Install Python packages
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Set permissions and create systemd service
sudo tee /etc/systemd/system/dash-app.service > /dev/null <<EOF
[Unit]
Description=Dash App Service
After=network.target

[Service]
User=$USER
WorkingDirectory=/home/$USER/overnight-map
ExecStart=/usr/bin/gunicorn -b 0.0.0.0:8080 main:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd, enable and start the app
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable dash-app
sudo systemctl start dash-app

echo "✅ Setup complete. App running at http://<your-external-ip>:8080"
