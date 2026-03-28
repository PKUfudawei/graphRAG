import json, os, requests, sys, time, fitz, io, threading

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
MODEL = "PaddleOCR-VL-1.5"

headers = {
    "Authorization": f"bearer {os.environ['PADDLE_TOKEN']}",
}

optional_payload = {
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useChartRecognition": False,
}


def trim_pdf(input_pdf, meta_pdf_path, trimmed_pdf_path):
    doc = fitz.open(input_pdf)
    
    meta_doc = fitz.open()
    meta_doc.insert_pdf(doc, from_page=0, to_page=0)
    meta_doc.save(meta_pdf_path)
    meta_doc.close()

    new_doc = fitz.open()
    cut_page = len(doc) - 1
    
    for i in range(1, len(doc)):
        text = doc[i].get_text().lower()
        if "acknowledgement" in text or 'acknowledgment' in text or 'references' in text:
            cut_page = i
            break

    new_doc.insert_pdf(doc, from_page=1, to_page=cut_page)
    new_doc.save(trimmed_pdf_path)
    new_doc.close()
    doc.close()


def submit_job(file_path):
    data = {
        "model": MODEL,
        "optionalPayload": json.dumps(optional_payload)
    }

    with open(file_path, "rb") as f:
        files = {"file": f}
        resp = requests.post(JOB_URL, headers=headers, data=data, files=files)

    assert resp.status_code == 200
    jobId = resp.json()["data"]["jobId"]
    return jobId


def wait_for_result(jobId):
    while True:
        r = requests.get(f"{JOB_URL}/{jobId}", headers=headers)
        assert r.status_code == 200
        data = r.json()["data"]
        state = data["state"]

        if state == "done":
            return data["resultUrl"]["jsonUrl"]
        elif state == "failed":
            print("Job failed:", data["errorMsg"])
            return None    
        time.sleep(1)


def download_and_merge(jsonl_url, final_md):
    resp = requests.get(jsonl_url)
    resp.raise_for_status()

    lines = resp.text.strip().split("\n")

    all_md = []
    stop = False

    for line in lines:
        if not line.strip() or stop:
            continue

        result = json.loads(line)["result"]
        for res in result["layoutParsingResults"]:
            if stop:
                continue
            md_text = res["markdown"]["text"]
            lower = md_text.lower()
            keywords = ['acknowledgement', 'acknowledgment', 'references']
            indices = [lower.find(kw) for kw in keywords]
            valid_indices = [idx for idx in indices if idx != -1]
            if valid_indices:
                cut_index = min(valid_indices)
                all_md.append(md_text[:cut_index])
                stop = True
                break

            all_md.append(md_text)

    with open(final_md, "w", encoding="utf-8") as f:
        f.write("\n".join(all_md))
        
semaphore = threading.Semaphore(5)
def process_pdf(pdf_path, md_path):
    if os.path.exists(md_path):
        return

    try:
        with semaphore:
            jobId = submit_job(pdf_path)
        json_url = wait_for_result(jobId)
        if json_url:
            download_and_merge(json_url, md_path)
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")


if __name__ == "__main__":
    max_workers = 8
    input_dir = "../data/papers"
    meta_dir = input_dir.replace('papers', 'meta_pdf')
    trim_dir = input_dir.replace('papers', 'main_pdf')
    meta_md_dir = input_dir.replace('papers', 'meta_markdown')
    main_md_dir = input_dir.replace('papers', 'main_markdown')
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(trim_dir, exist_ok=True)
    os.makedirs(meta_md_dir, exist_ok=True)
    os.makedirs(main_md_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    for file in tqdm(pdf_files, desc="Trimming PDFs", unit="file"):
        input_pdf = os.path.join(input_dir, file)
        trimmed_pdf = os.path.join(trim_dir, file)
        meta_pdf = os.path.join(meta_dir, file)
        trim_pdf(input_pdf, meta_pdf, trimmed_pdf)
        
    
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for file in pdf_files:
            meta_pdf = os.path.join(meta_dir, file)
            meta_md = os.path.join(meta_md_dir, file.replace(".pdf", ".md"))
            tasks.append(executor.submit(process_pdf, meta_pdf, meta_md))

        for _ in tqdm(as_completed(tasks), total=len(tasks), desc="OCR meta PDFs"):
            pass
        
    
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for file in pdf_files:
            main_pdf = os.path.join(trim_dir, file)
            main_md = os.path.join(main_md_dir, file.replace(".pdf", ".md"))

            tasks.append(executor.submit(process_pdf, main_pdf, main_md))

        for _ in tqdm(as_completed(tasks), total=len(tasks), desc="OCR main PDFs"):
            pass

