#test
def drug_features_to_text(drug) -> str:
    result = 'Drug: ' + drug['name'] + ', Drug Id: ' + drug['drugbank_id']
    + ', Type: ' + drug['type'] + "."
    if drug.get("categories"):
        result += "Categories:" +  ', '.join(drug['categories']) + "."
    if drug.get("atc_codes"):
        result += "ATC Codes:" +  ', '.join(drug['atc_codes']) + "."
    if drug.get("protein_binding"):
        result += "Protein binding:" +  ', '.join(drug['protein_binding']) + "."
    # TODO: finish rest of this
