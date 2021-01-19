sudo mkdir -p ~/.streamlit

sudo echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml