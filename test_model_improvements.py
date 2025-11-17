#!/usr/bin/env python3
"""
Quick test script to verify model improvements
Tests with just a few memes to check if viral model is better
"""

from meme_benchmarker import MemeBenchmarker
import pickle

# Test with just 5 popular memes
test_memes = ["doge", "pepe", "wojak", "chad", "sigma"]

print("🧪 Testing Model Improvements")
print("=" * 50)

benchmarker = MemeBenchmarker()

print("\n📊 Creating test dataset with 5 memes...")
benchmarker.create_meme_dataset(test_memes)

if len(benchmarker.meme_data) < 3:
    print("❌ Not enough data collected")
    exit(1)

print(f"\n✅ Collected data for {len(benchmarker.meme_data)} memes")

# Check the data
print("\n📈 Sample Data:")
for meme in benchmarker.meme_data[:3]:
    print(f"  {meme['meme_name']}: lifespan={meme['lifespan_days']:.1f} days, viral={meme['viral_score']:.3f}")

print("\n🧠 Training models on test data...")
success = benchmarker.train_ensemble_models()

if success:
    print("\n✅ Models trained successfully!")
    print("\n📊 Check the R² scores above - viral_potential should be positive now!")
else:
    print("\n❌ Training failed")

