# Customer Segmentation & Targeted Marketing

A Streamlit portfolio demo for customer segmentation using RFM analysis and K-Means clustering.

## Features
- Upload an existing `customer_segments.csv` file exported from the project notebook.
- Or upload raw retail transactions as CSV or Excel to calculate RFM features and train a K-Means model.
- KPI cards, segment distribution chart, monetary value chart, RFM scatter plots, segment summary, and marketing suggestions.
- Filter by customer segment and download the filtered customer table as CSV.

## Deploy on Streamlit Community Cloud
1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Choose **Create app**.
3. Select this repository, branch `main`, and main file path `app.py`.
4. Click **Deploy**.
5. When the app starts, upload your project CSV through the sidebar.

## Expected raw transaction columns
The raw data option expects columns equivalent to `InvoiceNo`, `CustomerID`, `Quantity`, `InvoiceDate`, and `UnitPrice`. Common variations in spaces and capitalization are supported.

## Run locally
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Notes
- The app expects you to upload a CSV/Excel file each time unless you later add a public, non-sensitive sample dataset to the repository.
- Do not commit confidential or personally identifiable customer data.
- Segment names inferred from raw data are heuristic and may not match manually named clusters in the original notebook.
- This is a portfolio demonstration, not a production marketing system.
