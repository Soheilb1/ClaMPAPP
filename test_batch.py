import pandas as pd
import time
from src.workflow import MedicalDiagnosisWorkflow

def test_on_dataset(file_path: str, text_column: str, sample_size: int = 30):
    """
    Test the pipeline on a dataset of clinical texts.
    """
    print(f"📥 Loading dataset from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{file_path}'.")
        return

    # پیدا کردن ستون حاوی متن
    if text_column not in df.columns:
        print(f"❌ Error: Column '{text_column}' not found.")
        print(f"Available columns: {list(df.columns)}")
        return

    # انتخاب یک نمونه کوچک برای تست اولیه
    df_test = df.head(sample_size).copy()
    
    print("⏳ Initializing the Hybrid LLM-XGBoost Pipeline...\n")
    workflow = MedicalDiagnosisWorkflow()
    
    success_count = 0
    
    for index, row in df_test.iterrows():
        clinical_text = str(row[text_column])
        print(f"\n▶️ Testing Sample #{index + 1}...")
        
        try:
            # یک وقفه کوتاه برای جلوگیری از خطای Rate Limit سرورهای OpenAI/Groq
            time.sleep(2) 
            
            response, probabilities = workflow.run(clinical_text)
            
            # استخراج کلمه کلیدی پیش‌بینی شده از متن پاسخ
            prediction = response.split('**')[1] if '**' in response else 'Unknown'
            
            print(f"✅ Success! Prediction: {prediction} | Probabilities: {probabilities}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error on Sample #{index + 1}: {e}")
            
    print("\n" + "="*50)
    print(f"🎯 TEST SUMMARY: {success_count} out of {len(df_test)} samples processed successfully.")
    print("="*50)

if __name__ == "__main__":
    # مسیر دقیق فایل دوسلدورف استاندارد
    DATASET_PATH = "data/dusseldorf_standard.csv" 
    
    # نکته مهم: نام ستونی که متن‌های بالینی در فایل دوسلدورف قرار دارند.
    # من پیش‌فرض "Patient_Note" گذاشتم، اما حتماً فایل CSV را چک کن که اسم این ستون دقیقاً چیست.
    TEXT_COLUMN_NAME = "text" 
    
    # تعداد ردیف‌هایی که می‌خواهید تست کنید (برای شروع همان ۵ عالی است)
    NUMBER_OF_SAMPLES_TO_TEST = 5 
    
    test_on_dataset(
        file_path=DATASET_PATH, 
        text_column=TEXT_COLUMN_NAME, 
        sample_size=NUMBER_OF_SAMPLES_TO_TEST
    )