# Turn on the server for embedding generation in the background
python  demo/backend/servers/embedding_generation/server.py --port 7862 &

# Turn on the server for retrieval in the background
python  demo/backend/servers/retrieval/server.py --port 7863 &