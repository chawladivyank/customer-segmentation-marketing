import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title='Customer Segmentation', page_icon='📊', layout='wide')
st.title('Customer Segmentation & Targeted Marketing')
st.caption('RFM analysis • K-Means clustering • Marketing insights')
st.info('Upload your existing customer_segments.csv, or upload the original retail transactions CSV/Excel file to calculate RFM and train K-Means.')

with st.sidebar:
    st.header('Data settings')
    mode = st.radio('Choose data type', ['Existing customer segments', 'Raw retail transactions'])
    upload = st.file_uploader('Upload CSV or Excel', type=['csv', 'xlsx', 'xls'])
    k = st.slider('K-Means clusters (raw data only)', 2, 8, 4)

def read_file(f):
    if f.name.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(f)
    return pd.read_csv(f)

def find_col(df, names):
    normalized = {str(c).strip().lower().replace(' ', '').replace('_', ''): c for c in df.columns}
    for name in names:
        key = name.lower().replace(' ', '').replace('_', '')
        if key in normalized:
            return normalized[key]
    return None

def existing_segments(df):
    df = df.copy()
    cust = find_col(df, ['CustomerID', 'Customer ID', 'customer_id'])
    seg = find_col(df, ['Segment', 'CustomerSegment', 'ClusterName', 'Cluster'])
    rec = find_col(df, ['Recency'])
    freq = find_col(df, ['Frequency', 'InvoiceCount'])
    mon = find_col(df, ['Monetary', 'TotalSpend', 'Sales', 'monetary_value'])
    if seg is None:
        return None, 'Could not find a segment column. Expected a column named Segment or ClusterName.'
    ren = {seg: 'Segment'}
    if cust: ren[cust] = 'CustomerID'
    if rec: ren[rec] = 'Recency'
    if freq: ren[freq] = 'Frequency'
    if mon: ren[mon] = 'Monetary'
    df = df.rename(columns=ren)
    if 'CustomerID' not in df: df['CustomerID'] = range(1, len(df) + 1)
    for col in ['Recency', 'Frequency', 'Monetary']:
        if col not in df: df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['Segment'] = df['Segment'].astype(str)
    return df, None

def train_raw(df, k):
    df = df.copy()
    cust = find_col(df, ['CustomerID', 'Customer ID'])
    invoice = find_col(df, ['InvoiceNo', 'Invoice No', 'Invoice'])
    qty = find_col(df, ['Quantity', 'Qty'])
    price = find_col(df, ['UnitPrice', 'Unit Price', 'Price'])
    date = find_col(df, ['InvoiceDate', 'Invoice Date', 'Date'])
    missing = [label for label, col in [('CustomerID', cust), ('InvoiceNo', invoice), ('Quantity', qty), ('UnitPrice', price), ('InvoiceDate', date)] if col is None]
    if missing: return None, 'Missing columns: ' + ', '.join(missing)
    df[cust] = df[cust].astype('string')
    df[qty] = pd.to_numeric(df[qty], errors='coerce')
    df[price] = pd.to_numeric(df[price], errors='coerce')
    df[date] = pd.to_datetime(df[date], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[cust, invoice, qty, price, date])
    df = df[(df[qty] > 0) & (df[price] > 0)]
    if df.empty: return None, 'No valid transactions after cleaning. Check your file.'
    df['Sales'] = df[qty] * df[price]
    snapshot = df[date].max().normalize() + pd.Timedelta(days=1)
    rfm = df.groupby(cust).agg(LastPurchase=(date, 'max'), Frequency=(invoice, 'nunique'), Monetary=('Sales', 'sum')).reset_index().rename(columns={cust: 'CustomerID'})
    rfm['Recency'] = (snapshot - rfm['LastPurchase'].dt.normalize()).dt.days
    rfm = rfm.drop(columns=['LastPurchase'])
    scaled = StandardScaler().fit_transform(rfm[['Recency', 'Frequency', 'Monetary']].astype(float))
    rfm['ClusterID'] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    profile = rfm.groupby('ClusterID')[['Recency', 'Frequency', 'Monetary']].mean()
    profile['score'] = -profile['Recency'].rank(pct=True) + profile['Frequency'].rank(pct=True) + profile['Monetary'].rank(pct=True)
    ranked = profile.sort_values('score', ascending=False).index.tolist()
    labels = {}
    if ranked: labels[ranked[0]] = 'VIP Customers'
    if len(ranked) > 1: labels[ranked[1]] = 'Loyal Customers'
    if len(ranked) > 2: labels[ranked[-1]] = 'At Risk'
    for cluster in ranked:
        if cluster not in labels: labels[cluster] = 'Potential Customers'
    rfm['Segment'] = rfm['ClusterID'].map(labels)
    return rfm, None

if upload is None:
    st.subheader('Start by uploading your data')
    st.markdown('**Existing segments:** upload `customer_segments.csv` from your notebook.\n\n**Raw transactions:** upload the original retail data containing `InvoiceNo`, `CustomerID`, `Quantity`, `InvoiceDate`, and `UnitPrice`.')
    st.stop()

try:
    source = read_file(upload)
except Exception as e:
    st.error(f'Could not read this file: {e}')
    st.stop()
if source.empty:
    st.error('The uploaded file is empty.')
    st.stop()
customers, error = existing_segments(source) if mode == 'Existing customer segments' else train_raw(source, k)
if error:
    st.error(error)
    st.stop()
customers = customers.drop_duplicates(subset=['CustomerID']).copy()
for col in ['Recency', 'Frequency', 'Monetary']:
    customers[col] = pd.to_numeric(customers[col], errors='coerce').fillna(0)

c1, c2, c3, c4 = st.columns(4)
c1.metric('Total customers', f'{customers.CustomerID.nunique():,}')
c2.metric('Total monetary value', f"£{customers.Monetary.sum():,.2f}")
c3.metric('Average value per customer', f"£{customers.Monetary.mean():,.2f}")
c4.metric('Number of segments', str(customers.Segment.nunique()))
st.divider()
left, right = st.columns(2)
with left:
    st.subheader('Customer distribution')
    counts = customers['Segment'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with right:
    st.subheader('Monetary value by segment')
    revenue = customers.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    revenue.plot(kind='bar', ax=ax)
    ax.set_ylabel('Monetary value')
    ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.subheader('Explore segments')
segments = sorted(customers['Segment'].unique())
selected = st.multiselect('Filter by segment', segments, default=segments)
filtered = customers[customers['Segment'].isin(selected)]
left, right = st.columns(2)
with left:
    st.markdown('**Recency vs Frequency**')
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, group in filtered.groupby('Segment'):
        ax.scatter(group['Recency'], group['Frequency'], label=name, alpha=.65)
    ax.set_xlabel('Recency in days (lower is more recent)')
    ax.set_ylabel('Frequency (unique invoices)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with right:
    st.markdown('**Frequency vs Monetary value**')
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, group in filtered.groupby('Segment'):
        ax.scatter(group['Frequency'], group['Monetary'], label=name, alpha=.65)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Monetary value')
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.subheader('Segment summary')
summary = filtered.groupby('Segment').agg(Customers=('CustomerID', 'nunique'), Avg_Recency=('Recency', 'mean'), Avg_Frequency=('Frequency', 'mean'), Avg_Monetary=('Monetary', 'mean'), Total_Monetary=('Monetary', 'sum')).reset_index()
st.dataframe(summary.round(2), use_container_width=True, hide_index=True)
st.subheader('Suggested marketing actions')
actions = {'VIP Customers': 'Offer exclusive rewards, early access, and personalized service.', 'Loyal Customers': 'Use loyalty points, repeat-purchase bundles, and referral offers.', 'Potential Customers': 'Test product recommendations and relevant offers to encourage repeat purchases.', 'At Risk': 'Consider a win-back campaign or feedback request.', 'At-Risk': 'Consider a win-back campaign or feedback request.'}
for segment in selected:
    with st.expander(segment): st.write(actions.get(segment, 'Review this group’s RFM profile and test a relevant campaign.'))
st.download_button('Download filtered customers (CSV)', filtered.to_csv(index=False).encode('utf-8'), 'customer_segments_filtered.csv', 'text/csv')
st.caption('Portfolio demo: segment labels generated from raw data are heuristic. Validate results before making business decisions.')
