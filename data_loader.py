import io
import json
import os
import boto3
from lxml import etree
from dotenv import load_dotenv

class DrugBankLoader:
    def __init__( self, aws_access_key_id: str = "key", aws_secret_access_key: str = "password",
        aws_region: str = "us-west-2", bucket_name: str = "drugbank-data-83", file_key: str = "full database.xml", chunk_size: int = 32 * 1024 * 1024) :
        self.bucket_name = bucket_name
        self.file_key = file_key
        self.chunk_size = chunk_size
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )

    def load(self, output_path="drugs.jsonl"):
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.file_key)
        stream = io.BufferedReader(response["Body"], buffer_size=self.chunk_size)
        
        ns = None
        
        with open(output_path, "w") as f:
            for event, elem in etree.iterparse(stream, events=("start", "end"), recover=True):
                if ns is None and event == "start":
                    ns = elem.tag.split("}")[0] + "}" if "}" in elem.tag else ""

                if event == "end" and elem.tag == f"{ns}drug" and elem.get("type"):
                    f.write(json.dumps(self._extract(elem, ns)) + "\n")
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]

        print(f"Done. Total: {count}")
        return count

    def _extract(elem, ns):
        def find(tag):
            return elem.findtext(f"{ns}{tag}")

        def polypeptide_id(e):
            p = e.find(f"{ns}polypeptide")
            return p.get("id") if p is not None else None

        return {
            "drugbank_id": elem.findtext(f"{ns}drugbank-id[@primary='true']") or find("drugbank-id"),
            "name":        find("name"),
            "type":        elem.get("type"),
            "atc_codes":   [e.get("code") for e in elem.findall(f".//{ns}atc-code")],
            "categories":  [c for e in elem.findall(f".//{ns}category") if (c := e.findtext(f"{ns}category"))],
            "targets":     [{"name": t.findtext(f"{ns}name"), "uniprot": polypeptide_id(t), "actions": [a.text for a in t.findall(f".//{ns}action")]} for t in elem.findall(f".//{ns}target")],
            "enzymes":     [{"name": e.findtext(f"{ns}name"), "uniprot": polypeptide_id(e)} for e in elem.findall(f".//{ns}enzyme")],
            "protein_binding": find("protein-binding"),
            "interactions": [{"drugbank_id": i.findtext(f"{ns}drugbank-id"), "name": i.findtext(f"{ns}name"), "description": i.findtext(f"{ns}description")} for i in elem.findall(f".//{ns}drug-interaction")],
        }
def main():
    loader = DrugBankLoader()
    loader.load("drugs.jsonl")

    with open("drugs.jsonl") as f:
        drugs = [json.loads(line) for line in f]

if __name__ == "__main__":
    main()
