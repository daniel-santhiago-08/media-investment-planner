mkdir -p ~/.streamlit/

mkdir -p  ~/Área de Trabalho/Daniel/MIP/media-investment-planner/teste/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml