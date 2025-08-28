# preprocess.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os
import sys
from tqdm import tqdm

def main():
    print("=" * 60)
    print("🧬 BioMed-QA Data Preprocessing")
    print("=" * 60)
    
    # --- 1. Check for dataset file ---
    dataset_files = [
        'PubMed Multi Label Text Classification Dataset.csv',
        'pubmed_dataset.csv',
        'pubmed_data.csv'
    ]
    
    dataset_file = None
    for filename in dataset_files:
        if os.path.exists(filename):
            dataset_file = filename
            break
    
    if not dataset_file:
        print("❌ Error: Dataset file not found!")
        print(f"   Please ensure one of these files exists in the current directory:")
        for filename in dataset_files:
            print(f"   - {filename}")
        print("\n💡 You can download the PubMed dataset from:")
        print("   https://www.kaggle.com/datasets/owaiskhan9654/pubmed-multilabel-text-classification")
        sys.exit(1)
    
    # --- 2. Load and inspect the dataset ---
    print(f"📂 Loading dataset: {dataset_file}")
    try:
        df = pd.read_csv(dataset_file)
        print(f"✅ Dataset loaded successfully!")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)
    
    # --- 3. Data cleaning and validation ---
    print("\n🧹 Cleaning data...")
    
    # Check for required columns
    required_cols = ['abstractText']
    if 'articleTitle' not in df.columns:
        if 'title' in df.columns:
            df['articleTitle'] = df['title']
        else:
            df['articleTitle'] = 'Untitled'
            print("⚠️  Warning: No title column found, using 'Untitled'")
    
    if 'abstractText' not in df.columns:
        if 'abstract' in df.columns:
            df['abstractText'] = df['abstract']
        elif 'text' in df.columns:
            df['abstractText'] = df['text']
        else:
            print("❌ Error: No abstract text column found!")
            print(f"   Available columns: {list(df.columns)}")
            sys.exit(1)
    
    # Clean the data
    initial_count = len(df)
    df['abstractText'] = df['abstractText'].astype(str).str.strip().str.replace('\n', ' ', regex=False)
    df['articleTitle'] = df['articleTitle'].astype(str).str.strip()
    
    # Remove entries with very short abstracts (likely not useful)
    df = df[df['abstractText'].str.len() >= 50]
    df = df.dropna(subset=['abstractText'])
    
    print(f"   Removed {initial_count - len(df)} invalid entries")
    print(f"   Remaining entries: {len(df)}")
    
    # --- 4. Create sample ---
    # Adjust sample size based on available resources
    max_sample_size = min(10000, len(df))  # Increased from 5000
    sample_size = int(input(f"\n📊 Enter sample size (max {max_sample_size}, recommended 5000): ") or 5000)
    sample_size = min(sample_size, max_sample_size)
    
    print(f"📦 Creating sample of {sample_size} abstracts...")
    df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    # Show sample statistics
    avg_length = df_sample['abstractText'].str.len().mean()
    print(f"   Average abstract length: {avg_length:.0f} characters")
    
    # --- 5. Initialize the sentence transformer model ---
    print("\n🤖 Loading sentence transformer model...")
    print("   This may take a few minutes on first run...")
    
    model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    try:
        embedder = SentenceTransformer(model_name)
        print("✅ Model loaded successfully!")
        print(f"   Model: {model_name}")
        print(f"   Embedding dimension: {embedder.get_sentence_embedding_dimension()}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Trying fallback model...")
        try:
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Fallback model loaded (all-MiniLM-L6-v2)")
        except Exception as e2:
            print(f"❌ Error loading fallback model: {e2}")
            sys.exit(1)
    
    # --- 6. Compute embeddings ---
    print(f"\n🔄 Computing embeddings for {len(df_sample)} abstracts...")
    print("   This is the most time-consuming step...")
    
    batch_size = 32  # Process in batches to avoid memory issues
    
    try:
        # Enable progress bar and batch processing
        corpus_embeddings = embedder.encode(
            df_sample['abstractText'].tolist(),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True
        )
        
        # Convert to numpy for saving
        corpus_embeddings_np = corpus_embeddings.cpu().numpy()
        print("✅ Embeddings computed successfully!")
        print(f"   Embedding shape: {corpus_embeddings_np.shape}")
        
    except Exception as e:
        print(f"❌ Error computing embeddings: {e}")
        print("💡 Try reducing the sample size or check available memory")
        sys.exit(1)
    
    # --- 7. Save processed data ---
    print("\n💾 Saving processed data...")
    
    try:
        # Save the processed DataFrame
        output_csv = 'pubmed_sample_processed.csv'
        df_sample.to_csv(output_csv, index=False)
        print(f"✅ Saved processed data: {output_csv}")
        
        # Save the embeddings
        output_pkl = 'corpus_embeddings.pkl'
        with open(output_pkl, 'wb') as f:
            pickle.dump(corpus_embeddings_np, f)
        print(f"✅ Saved embeddings: {output_pkl}")
        
        # Save processing metadata
        metadata = {
            'sample_size': len(df_sample),
            'embedding_dim': corpus_embeddings_np.shape[1],
            'model_name': model_name,
            'average_abstract_length': avg_length
        }
        
        with open('preprocessing_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        print("✅ Saved metadata: preprocessing_metadata.pkl")
        
    except Exception as e:
        print(f"❌ Error saving files: {e}")
        sys.exit(1)
    
    # --- 8. Summary ---
    print("\n" + "=" * 60)
    print("🎉 Preprocessing completed successfully!")
    print("=" * 60)
    print(f"📊 Processed {len(df_sample)} abstracts")
    print(f"🎯 Embedding dimension: {corpus_embeddings_np.shape[1]}")
    print(f"💾 Files created:")
    print(f"   - {output_csv}")
    print(f"   - {output_pkl}")
    print(f"   - preprocessing_metadata.pkl")
    print(f"\n🚀 Ready to run the Streamlit app:")
    print(f"   streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()