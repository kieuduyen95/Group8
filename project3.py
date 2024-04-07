#IMPORT LIBRARIES
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import squarify
from io import BytesIO
from datetime import datetime
from underthesea import word_tokenize, pos_tag, sent_tokenize
import jieba
import re
import string
from wordcloud import WordCloud
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx as get_report_ctx

# LOADING DATA

# Load raw data:
df_1rst = pd.read_csv('data/OnlineRetail_1rst.csv', sep=",", encoding='latin')
df_2nd = pd.read_csv('data/OnlineRetail_2nd.csv', sep=",", encoding='latin')
frames = [df_1rst,df_2nd]
df_raw = pd.concat(frames)

# Xử lý raw data:
df_raw['InvoiceDate'] = pd.to_datetime(df_raw['InvoiceDate'], format='%d-%m-%Y %H:%M')
df_raw['GrossSale'] =  df_raw['Quantity'] * df_raw['UnitPrice']
#df_raw['CustomerID'] = df_raw['CustomerID'].astype('str')

# Tạo dataframes chỉ chứa KH định danh và đơn hàng không bị trả hoặc sale:
df_cust = df_raw.dropna(subset=['CustomerID'])
df_cust = df_raw.loc[(df_raw['Quantity'] > 0) & (df_raw['UnitPrice'] > 0)]
df_cust['CustomerID'] = df_cust['CustomerID'].apply(lambda x: str(x).replace('.0',''))

df_new = df_cust.groupby(['InvoiceNo', 'InvoiceDate', 'CustomerID', 'Country'])['GrossSale'].sum().reset_index()
df_new['CustomerID'] = df_new['CustomerID'].apply(lambda x: str(x).replace('.0',''))

# Load file đã xử lý
rfm_agg = pd.read_csv('data/rfm_agg.csv')
cust_rfm = pd.read_csv('data/cust_rfm.csv')
rfm = pd.read_csv('data/rfm.csv')

cust_rfm['CustomerID'] = cust_rfm['CustomerID'].apply(lambda x: str(x).replace('.0',''))

STOP_WORD_FILE = 'data/stopwords-en.txt'
with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
    stop_words = file.read()
stop_words = stop_words.split('\n')

# USING MENU
st.title("Customer Segmentation")
menu = ["Trang chủ", "Tổng quan kinh doanh", "Công cụ phân nhóm", "Phân tích khách hàng"]
choice = st.sidebar.selectbox('Trang chủ', menu)
if choice == "Trang chủ":
    st.image('data/pics/Customer-Segmentation.jpg')

elif choice =='Tổng quan kinh doanh':
    st.image('data/pics/customer-segmentation.webp')
    st.subheader("TỔNG QUAN TÌNH HÌNH KINH DOANH ONLINE")
    st.write("##### 1. Tổng quan đơn hàng:")
    
    # Đếm số lượng đơn hàng
    a = len(df_raw['InvoiceNo'].unique())
    b = len(df_raw.loc[df_raw['Quantity'] <= 0]['InvoiceNo'].unique())
    c = len(df_raw.loc[df_raw['UnitPrice'] <= 0]['InvoiceNo'].unique())
    st.dataframe(pd.DataFrame({'Đơn hàng': ['Tổng', 'Bị trả','Bị điều chỉnh hoặc sale', 'Thực tế'], 'Số lượng': [a,b,c,a-b-c]}))
    
    # Khách hàng vãng lai
    bill_vanglai = len(df_raw.loc[df_raw['CustomerID'].isna()]['InvoiceNo'].unique())*100/len(df_raw['InvoiceNo'].unique())
    st.write("Tỷ lệ đơn hàng từ KH vãng lai: ",round(bill_vanglai,2),"%")
    st.write("Tỷ lệ đơn hàng từ KH định danh: ",100 - round(bill_vanglai,2),"%")
    st.write('Giao dịch được tổng hợp từ {} đến {}'.format(df_raw['InvoiceDate'].min(), df_raw['InvoiceDate'].max()))
    st.write('Trong đó số lượng KH được định danh là {:,} '.format(len(df_cust['CustomerID'].unique())))
    
    # Thống kê số lượng KH và đơn hàng theo quốc gia
    df_country = df_cust.groupby('Country').agg({'CustomerID': lambda x: len(x.unique()), 'InvoiceNo': lambda x: len(x.unique())}).reset_index()
    df_country.columns = ['Country', 'Số lượng KH', 'Số lượng đơn hàng']
    df_country.sort_values(by = 'Số lượng KH', ascending = False, inplace = True)
    df_country.reset_index(drop=True, inplace=True)
    st.write('Thống kê đơn hàng theo quốc gia:')
    st.dataframe(df_country.head())

    # Biểu đồ số lượng KH theo quốc gia
    df_bar = df_country.head()
    colors = ['salmon', 'limegreen','gold', 'pink','skyblue']
    sorted_indices = sorted(range(len(df_bar['Số lượng KH'])), key=lambda i: df_bar['Số lượng KH'][i], reverse=False)
    sorted_countries = [df_bar['Country'][i] for i in sorted_indices]
    sorted_num_countries = [df_bar['Số lượng KH'][i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_countries, sorted_num_countries, color=colors)
    plt.xlabel('Số lượng KH')
    plt.ylabel('Country')
    plt.title('Top 5 quốc gia có nhiều khách hàng nhất')
    plt.tight_layout()
    st.pyplot(plt)

    # Biểu đồ số lượng đơn hàng theo quốc gia
    #colors = ['salmon', 'limegreen','gold', 'pink','skyblue']
    sorted_indices1 = sorted(range(len(df_bar['Số lượng đơn hàng'])), key=lambda i: df_bar['Số lượng đơn hàng'][i], reverse=False)
    sorted_countries1 = [df_bar['Country'][i] for i in sorted_indices1]
    sorted_num_countries1 = [df_bar['Số lượng đơn hàng'][i] for i in sorted_indices1]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_countries1, sorted_num_countries1, color=colors)
    plt.xlabel('Số lượng đơn hàng')
    plt.ylabel('Country')
    plt.title('Top 5 quốc gia có nhiều đơn hàng nhất')
    plt.tight_layout()
    st.pyplot(plt)

    st.write('#### WordCloud top sản phẩm của PK VIPs:')
    st.image('data/pics/VIPs.png')
    st.write('#### WordCloud top sản phẩm của PK Big Spender:')
    st.image('data/pics/bigspender.png')
    st.write('#### WordCloud top sản phẩm của PK New Customer:')
    st.image('data/pics/newcust.png')

elif choice == "Công cụ phân nhóm":
    st.image('data/pics/cust.png')
    st.write('### Manual Segmentation')
    st.write("""Customer Segmentation là một công cụ mạnh mẽ giúp doanh nghiệp hiểu sâu hơn về khách hàng của họ và cách tùy chỉnh chiến lược tiếp thị
                            Đây là một bước không thể thiếu để đảm bảo rằng bạn đang tiếp cận và phục vụ mọi nhóm khách hàng một cách hiệu quả""")
    st.write("""Tiêu chí phân loại Khách hàng:""")
    st.write("""
            + VIPs: Khách hàng có lượng chi tiêu lớn, tần suất tiêu thụ thường xuyên, và vừa shopping gần đây
            + BIG Spender: Khách hàng chi tiêu lớn, nhưng khác VIPs ở điểm không tiêu thụ thường xuyên bằng
            + LOYAL: Khách hàng thường đến, và vẫn đang acitve (đến gần đây), mức độ chi tiêu kém hơn VIPs
            + NEWCUST: Khách hàng mới đến gần đây, chưa quan tâm đến mức độ chi tiêu
            + LIGHT: Khách hàng có lượng chi tiêu ít nhất trong nhóm đang active, nhưng vẫn thường đến
            + LOST: Khách hàng quá lâu chưa đến, và thường chi tiêu ít
            + REGULARS: Nhóm còn lại, thường ở mức trung bình ở 3 khía cạnh M, F, R
            """)
    st.write('### Danh sách khách hàng hiện hữu và thông tin mua sắm chi tiết')
    st.dataframe(df_cust.head())
    st.write('### Bảng giá trị trung bình Recency-Frequency-Monetary theo phân cụm khách hàng')
    st.dataframe(rfm_agg)

    # Show biểu đồ phân cụm
    st.write('### TreeMap')
    st.image('data/pics/RFM Segments.png')
    st.write('### Scatter Plot (RFM)')
    st.image('data/pics/Scatter Segments.png')

elif choice=='Phân tích khách hàng':
    st.image('data/pics/seg.jpg')
    st.subheader("PHÂN TÍCH KHÁCH HÀNG")
    type = st.radio("### Nhập thông tin khách hàng", options=["Mã khách hàng", "Hành vi mua sắm của khách hàng", "Upload file"])
    if type == "Mã khách hàng":
        st.subheader("Mã khách hàng")
        # Tạo điều khiển để người dùng nhập và chọn nhiều mã khách hàng từ danh sách gợi ý
        st.markdown("**Có thể nhập và chọn nhiều mã khách hàng từ danh sách gợi ý**")

        all_ids = df_cust['CustomerID'].unique()
        # Chọn nhiều ID từ danh sách
        selected_ids = st.multiselect("Chọn ID:", all_ids)
        # In ra danh sách ID đã chọn
        st.write("#### Bạn đã chọn các KH sau:")
        st.write(selected_ids)

        if any(id in df_cust['CustomerID'].values for id in selected_ids):

            # Đề xuất khách hàng thuộc cụm nào
            df_cust_rfm = cust_rfm[cust_rfm['CustomerID'].isin(selected_ids)].sort_values(['CustomerID'], ascending= False, ignore_index= True)
            st.write(f"#### Khách hàng đã chọn thuộc nhóm")
            st.dataframe(df_cust_rfm[['CustomerID', 'RFM_Level', 'RFM_Segment', 'Recency', 'Frequency', 'Monetary']])
            filtered_df_new = df_new[df_new['CustomerID'].isin(selected_ids)].sort_values(['CustomerID', 'InvoiceDate'], ascending= False, ignore_index= True)
            #st.dataframe(filtered_df_new)

            st.write("#### Khách hàng đã từng mua sắm ở:")
            country = filtered_df_new.groupby(['CustomerID', 'Country'])['Country'].value_counts().reset_index(drop=True)
            country.columns = ['CustomerID', 'Country', 'Số lần mua hàng']
            st.dataframe(country)
            
            st.write("#### Khoảng chi tiêu ($):")
            grosssale = filtered_df_new.groupby('CustomerID').agg({'GrossSale': ['min', 'max', 'sum']}).reset_index()
            grosssale.columns = ['CustomerID', 'Min', 'Max', 'Total']
            st.dataframe(grosssale)
            
            st.write("#### Thông tin mua hàng sắp xếp theo lần gần nhất:")
            st.dataframe(filtered_df_new)

            def cust_top_product(selected_ids):
                df = df_cust[df_cust['CustomerID'].isin(selected_ids)]
                df_grouped = df.groupby('Description')['StockCode'].count().reset_index()
                df_grouped = df_grouped.rename(columns={'StockCode': 'Count'})
                df_top = df_grouped.sort_values(by='Count', ascending=False).head(5)
                #df_top = df.groupby(['Description','UnitPrice'])['StockCode'].value_counts().sort_values(ascending=False).head(20).reset_index(drop=True)
                return df_top
            def text_underthesea(text):
                products_wt = text.str.lower().apply(lambda x: word_tokenize(x, format="text"))
                products_name_pre = [[text for text in set(x.split())] for x in products_wt]
                products_name_pre = [[re.sub('[0-9]+','', e) for e in text] for text in products_name_pre]
                products_name_pre = [[t.lower() for t in text if not t in ['', ' ', ',', '.', '...', '-',':', ';', '?', '%', '_%' , '(', ')', '+', '/', 'g', 'ml']]
                                    for text in products_name_pre] # ký tự đặc biệt
                products_name_pre = [[t for t in text if not t in stop_words] for text in products_name_pre] # stopword
                return products_name_pre
            def wcloud_visualize(input_text):
                flat_text = [word for sublist in input_text for word in sublist]
                text = ' '.join(flat_text)
                wc = WordCloud(
                                background_color='gold',
                                colormap="ocean_r",
                                max_words=50,
                                width=1600,
                                height=900,
                                max_font_size=400)
                wc.generate(text)
                # Save the WordCloud image
                wc.to_file("wordcloud.png")

                # Display the saved image using st.image
                st.image("wordcloud.png")

            top_products = cust_top_product(selected_ids)
            #st.dataframe(top_products)
            st.write("#### Top 5 đơn hàng được mua nhiều nhất của các KH được chọn:")
            st.dataframe(top_products.head())
            st.write("#### Word Cloud Visualization:")
            sample_text = top_products['Description']
            processed_text = text_underthesea(sample_text)
            wcloud_visualize(processed_text)

        else:
            # Không có khách hàng
            st.write("Vui lòng chọn ID ở khung trên :rocket:")


    elif type == "Hành vi mua sắm của khách hàng":
        # Nếu người dùng chọn nhập thông tin khách hàng vào dataframe có 3 cột là Recency, Frequency, Monetary
        st.write("##### 2. Thông tin khách hàng")
        # Tạo điều khiển table để người dùng nhập thông tin khách hàng trực tiếp trên table
        st.write("Nhập thông tin khách hàng")

        # Loop to get input from the user for each customer
            # Get input using sliders
        # Tạo DataFrame rỗng
        df_customer = pd.DataFrame(columns=["Recency", "Frequency", "Monetary"])

        # Lặp qua 5 khách hàng
        for i in range(2):
            st.write(f"Khách hàng {i+1}")
            
            # Sử dụng sliders để nhập giá trị cho Recency, Frequency và Monetary
            recency = st.slider("Recency", 1, 365, 100, key=f"recency_{i}")
            frequency = st.slider("Frequency", 1, 50, 5, key=f"frequency_{i}")
            monetary = st.slider("Monetary", 1, 1000, 100, key=f"monetary_{i}")
            
            # Thêm dữ liệu nhập vào DataFrame
            df_customer = df_customer.append({"Recency": recency, "Frequency": frequency, "Monetary": monetary}, ignore_index=True)

        # Hiển thị DataFrame
        st.dataframe(df_customer)
                    
        # Create labels for Recency, Frequency, Monetary
        r_labels = range(4, 0, -1) #số ngày tính từ lần cuối mua hàng lớn thì gán nhãn nhỏ, ngược lại thì nhãn lớn
        f_labels = range(1, 5)
        m_labels = range(1, 5)

        # Assign these labels to 4 equal percentile groups
        r_groups = pd.qcut(df_customer['Recency'].rank(method = 'first'), q = 4, labels = r_labels)
        f_groups = pd.qcut(df_customer['Frequency'].rank(method = 'first'), q = 4, labels = f_labels)
        m_groups = pd.qcut(df_customer['Monetary'].rank(method = 'first'), q = 4, labels = m_labels)

        # Create new columns R, F, M
        df_customer = df_customer.assign(R = r_groups.values, F = f_groups.values, M = m_groups.values)
        def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
        df_customer['RFM_Segment'] = df_customer.apply(join_rfm, axis=1)
        # Calculate RFM_Score
        df_customer['RFM_Score'] = df_customer[['R', 'F', 'M']].sum(axis=1)

        def rfm_level(df):
        # Check for special "STARS" and "NEW" condition first
            if df['RFM_Score'] == 12:
                return "VIPs"
        # Then check for other conditons
            elif df['M'] == 4 and df['F'] != 4 and df['R'] != 4: # F=4 & R=4 --> VIPs
                return "BIG SPENDER"
            elif df['F'] >= 3 and df['R'] >= 3: # KH thường đến, ko quan tâm M lớn nhỏ
                return "LOYAL"
            elif df['R'] == 4 and df['F'] == 1: # KH mới đến lần đầu, ko quan tâm M
                return "NEWCUST"
            elif df['M'] < 3 and df['R'] != 1 and df['F'] < 3: #nếu R = 1 thì thành LOST
                return "LIGTH"
            elif df['R'] == 1 and df['M'] == 1: # mua xa nhưng chi nhiều thì k đưa về nhóm lost
                return "LOST"
            else:
                return "REGULARS"

        # Create a new column RFM_Level
        df_customer['RFM_Level'] = df_customer.apply(rfm_level, axis=1)
        st.dataframe(df_customer)

    elif type == "Upload file":
        st.subheader("Upload file")
        uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                # Đọc file dữ liệu
                df_upload = pd.read_csv(uploaded_file)
                st.write(df_upload)
        submitted_project1 = st.button("Phân nhóm khách hàng")
        if submitted_project1:
            # Hiển thị DataFrame
            #st.dataframe(df_upload)

            # Create labels for Recency, Frequency, Monetary
            r_labels = range(4, 0, -1) #số ngày tính từ lần cuối mua hàng lớn thì gán nhãn nhỏ, ngược lại thì nhãn lớn
            f_labels = range(1, 5)
            m_labels = range(1, 5)

            # Assign these labels to 4 equal percentile groups
            r_groups = pd.qcut(df_upload['Recency'].rank(method = 'first'), q = 4, labels = r_labels)
            f_groups = pd.qcut(df_upload['Frequency'].rank(method = 'first'), q = 4, labels = f_labels)
            m_groups = pd.qcut(df_upload['Monetary'].rank(method = 'first'), q = 4, labels = m_labels)

            # Create new columns R, F, M
            df_customer = df_upload.assign(R = r_groups.values, F = f_groups.values, M = m_groups.values)
            def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
            df_customer['RFM_Segment'] = df_customer.apply(join_rfm, axis=1)
            # Calculate RFM_Score
            df_customer['RFM_Score'] = df_customer[['R', 'F', 'M']].sum(axis=1)

            def rfm_level(df):
            # Check for special "STARS" and "NEW" condition first
                if df['RFM_Score'] == 12:
                    return "VIPs"
            # Then check for other conditons
                elif df['M'] == 4 and df['F'] != 4 and df['R'] != 4: # F=4 & R=4 --> VIPs
                    return "BIG SPENDER"
                elif df['F'] >= 3 and df['R'] >= 3: # KH thường đến, ko quan tâm M lớn nhỏ
                    return "LOYAL"
                elif df['R'] == 4 and df['F'] == 1: # KH mới đến lần đầu, ko quan tâm M
                    return "NEWCUST"
                elif df['M'] < 3 and df['R'] != 1 and df['F'] < 3: #nếu R = 1 thì thành LOST
                    return "LIGTH"
                elif df['R'] == 1 and df['M'] == 1: # mua xa nhưng chi nhiều thì k đưa về nhóm lost
                    return "LOST"
                else:
                    return "REGULARS"

            # Create a new column RFM_Level
            df_customer['RFM_Level'] = df_customer.apply(rfm_level, axis=1)
            st.dataframe(df_customer)
            











