# Use a standard Python 3.9 image
FROM python:3.9

# Set up a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory inside the container
WORKDIR /app

# Copy and install requirements
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all your project files (.py, .md, etc.) into the container
COPY --chown=user . /app

# --- THIS IS THE CRITICAL NEW SECTION ---
# 1. Run the script to create embeddings and sentences
#    This will create the 'knowledge_base/embeddings.npy' and 'knowledge_base/sentences.txt' files
RUN python create_embeddings.py

# 2. Run the script to build the FAISS index
#    This will read the files above and create 'knowledge_base/faiss.index'
RUN python build_index.py
# --- END NEW SECTION ---

# Expose the port (7860 is standard for Hugging Face Spaces)
EXPOSE 7860

# --- THIS IS THE FIXED COMMAND ---
# Use Gunicorn to run your Flask app ('app:app' means the 'app' object inside 'app.py')
# This replaces the incorrect 'uvicorn' command.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]