import requests, time, json, os, csv
import xml.etree.ElementTree as ET
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_meta_data(query, size=250):
    metas = []

    params = {
        "q": query,
        "size": size,
        "fields": "titles,arxiv_eprints,dois,files,control_number"
    }

    url = "https://inspirehep.net/api/literature"

    while True:
        r = requests.get(url, params=params)
        data = r.json()
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            break

        metas.extend(hits)
        print(f"Fetched: {len(metas)}")

        next_url = data.get("links", {}).get("next")
        if not next_url:
            break

        url = next_url
        params = None
        time.sleep(0.5)

    return metas


def doi_to_arxiv(doi, max_retires=3):
    for _ in range(max_retires):
        try:
            url = f"http://export.arxiv.org/api/query?search_query=doi:{doi}"
            r = requests.get(url, timeout=10)
            root = ET.fromstring(r.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entry = root.find("atom:entry", ns)
            if entry is None:
                return None
            return entry.find("atom:id", ns).text.split("/")[-1]
        except:
            time.sleep(0.5)
    return None


def download_pdf(doi=None, arxiv=None, output_dir='../data/papers'):
    if doi:
        try:
            url = f"https://doi.org/{doi}"
            r = requests.get(url, timeout=20, allow_redirects=True)
            if r.status_code == 200 and "pdf" in r.headers.get("content-type", "").lower():
                path = os.path.join(output_dir, f"{doi}.pdf")
                with open(path, "wb") as f:
                    f.write(r.content)
                return True
        except:
            pass

    if arxiv:
        try:
            url = f"https://arxiv.org/pdf/{arxiv}.pdf"
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                path = os.path.join(output_dir, f"{arxiv}.pdf")
                with open(path, "wb") as f:
                    f.write(r.content)
                return True
        except:
            pass

    return False


def process_one(meta, output_dir):
    metadata = meta['metadata']
    title = metadata.get("titles", [{}])[0].get("title", "UNKNOWN")
    doi = metadata['dois'][0]['value'] if 'dois' in metadata else None

    if 'arxiv_eprints' in metadata:
        arxiv = metadata['arxiv_eprints'][0]['value']
    elif doi:
        arxiv = doi_to_arxiv(doi)
    else:
        arxiv = None

    success = download_pdf(doi=doi, arxiv=arxiv, output_dir=output_dir)
    return success, title, doi, arxiv


def main():
    queries = {
        "ATLAS": 'collaboration:"ATLAS" AND collection:Published',
        "CMS": 'collaboration:"CMS" AND collection:Published',
    }
    os.makedirs('../data/papers', exist_ok=True)
    for k, v in queries.items():
        metas = fetch_meta_data(query=v)
        with open(f'../data/{k}.json', "w", encoding="utf-8") as f:
            json.dump(metas, f, ensure_ascii=False, indent=2)

    for k in queries:
        with open(f"../data/{k}.json", "r", encoding="utf-8") as f:
            metas = json.load(f)

        failed_file = f'../data/failed_{k}.csv'
        with open(failed_file, "w", encoding="utf-8", newline="") as ff:
            writer = csv.writer(ff)
            writer.writerow(["Title", "DOI", "arXiv"])

            print(f"\nStart downloading {k}, total: {len(metas)}")
            results = []
            max_workers = 16

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_one, m, '../data/papers') for m in metas]
                for f in tqdm(as_completed(futures), total=len(futures), desc=k):
                    success, title, doi, arxiv = f.result()
                    results.append((success, title, doi, arxiv))
                    if not success:
                        writer.writerow([title, doi, arxiv])

        success = sum(1 for r in results if r[0])
        print(f"{k} done: {success}/{len(results)} downloaded")
        print(f"All failed DOI/arXiv IDs saved to {failed_file}")


if __name__ == "__main__":
    main()
