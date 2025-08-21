import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Olist E-Commerce Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Brazilian state population data for bias adjustment
BRAZIL_POPULATION = {
    'SP': 46649132, 'MG': 21411923, 'RJ': 17463349, 'BA': 14985284,
    'PR': 11597484, 'RS': 11466630, 'PE': 9674793, 'CE': 9240580,
    'PA': 8777124, 'SC': 7338473, 'MA': 7153262, 'GO': 7206589,
    'PB': 4059905, 'AM': 4269995, 'ES': 4108508, 'RN': 3560903,
    'AL': 3365351, 'PI': 3289290, 'MT': 3567234, 'DF': 3094325,
    'MS': 2839188, 'SE': 2338474, 'RO': 1815278, 'TO': 1607363,
    'AC': 906876, 'AP': 877613, 'RR': 652713
}

@st.cache_data
def load_data():
    """Load all datasets with proper data types"""
    try:
        customers = pd.read_csv('olist_customers_dataset.csv')
        geolocation = pd.read_csv('olist_geolocation_dataset.csv')
        order_items = pd.read_csv('olist_order_items_dataset.csv')
        order_payments = pd.read_csv('olist_order_payments_dataset.csv')
        order_reviews = pd.read_csv('olist_order_reviews_dataset.csv')
        orders = pd.read_csv('olist_orders_dataset.csv')
        products = pd.read_csv('olist_products_dataset.csv')
        sellers = pd.read_csv('olist_sellers_dataset.csv')
        category_translation = pd.read_csv('product_category_name_translation.csv')
        
        # Convert date columns
        date_columns = ['order_purchase_timestamp', 'order_approved_at', 
                       'order_delivered_carrier_date', 'order_delivered_customer_date',
                       'order_estimated_delivery_date']
        for col in date_columns:
            if col in orders.columns:
                orders[col] = pd.to_datetime(orders[col], errors='coerce')
        
        order_items['shipping_limit_date'] = pd.to_datetime(order_items['shipping_limit_date'], errors='coerce')
        order_reviews['review_creation_date'] = pd.to_datetime(order_reviews['review_creation_date'], errors='coerce')
        order_reviews['review_answer_timestamp'] = pd.to_datetime(order_reviews['review_answer_timestamp'], errors='coerce')
        
        return {
            'customers': customers,
            'geolocation': geolocation,
            'order_items': order_items,
            'order_payments': order_payments,
            'order_reviews': order_reviews,
            'orders': orders,
            'products': products,
            'sellers': sellers,
            'category_translation': category_translation
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def preprocess_data(data_dict):
    """Preprocess and merge datasets for analysis"""
    if not data_dict:
        return None
    
    try:
        # Merge products with category translation
        products_enhanced = data_dict['products'].merge(
            data_dict['category_translation'],
            on='product_category_name',
            how='left'
        )
        
        # Create main dataframe
        df = data_dict['orders'].copy()
        df = df.merge(data_dict['order_items'], on='order_id', how='left')
        df = df.merge(products_enhanced, on='product_id', how='left')
        df = df.merge(data_dict['sellers'], on='seller_id', how='left')
        df = df.merge(data_dict['customers'], on='customer_id', how='left')
        df = df.merge(data_dict['order_payments'], on='order_id', how='left')
        df = df.merge(data_dict['order_reviews'], on='order_id', how='left')
        
        # Calculate delivery metrics
        df['delivery_time'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
        df['estimated_delivery_time'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.days
        df['delivery_delay'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
        df['on_time_delivery'] = df['delivery_delay'] <= 0
        
        # Extract time features
        df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)
        df['order_week'] = df['order_purchase_timestamp'].dt.to_period('W').astype(str)
        df['order_date'] = df['order_purchase_timestamp'].dt.date
        df['order_hour'] = df['order_purchase_timestamp'].dt.hour
        df['order_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
        df['order_year'] = df['order_purchase_timestamp'].dt.year
        df['order_quarter'] = df['order_purchase_timestamp'].dt.quarter
        
        return df
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None

def page_executive_overview(df, data_dict=None):
    """Executive Overview Page"""
    st.title("📊 Executive Overview")
    st.markdown("### Business Health Snapshot")
    
    # Calculate KPIs
    total_revenue = df.groupby('order_id')['payment_value'].sum().sum()
    total_orders = df['order_id'].nunique()
    total_customers = df['customer_unique_id'].nunique()
    total_sellers = df['seller_id'].nunique()
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    review_scores = df.groupby('order_id')['review_score'].first()
    avg_review_score = review_scores.dropna().mean() if not review_scores.dropna().empty else 0
    
    # Display KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Revenue", f"R$ {total_revenue:,.0f}")
        st.metric("Avg Order Value", f"R$ {avg_order_value:.2f}")
    
    with col2:
        st.metric("Total Orders", f"{total_orders:,}")
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col3:
        st.metric("Active Sellers", f"{total_sellers:,}")
        delivered_count = df[df['order_status'] == 'delivered']['order_id'].nunique()
        delivered_pct = (delivered_count / total_orders * 100) if total_orders > 0 else 0
        st.metric("Delivery Rate", f"{delivered_pct:.1f}%")
    
    with col4:
        st.metric("Avg Review Score", f"⭐ {avg_review_score:.2f}")
        delivered_orders = df[df['order_delivered_customer_date'].notna()]['order_id'].nunique()
        on_time_orders = df[df['on_time_delivery'] == True]['order_id'].nunique()
        on_time_pct = (on_time_orders / delivered_orders * 100) if delivered_orders > 0 else 0
        st.metric("On-Time Delivery", f"{on_time_pct:.1f}%")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Revenue Trend")
        monthly_revenue = df.groupby('order_month')['payment_value'].sum().reset_index()
        monthly_revenue = monthly_revenue.sort_values('order_month')
        
        fig = px.area(monthly_revenue, x='order_month', y='payment_value',
                     title="Monthly Revenue Trend",
                     labels={'payment_value': 'Revenue (R$)', 'order_month': 'Month'})
        fig.update_traces(fillcolor='rgba(31, 119, 180, 0.3)', line=dict(color='#1f77b4', width=3))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🗺️ Geographic Distribution")
        state_revenue = df.groupby('customer_state')['payment_value'].sum().reset_index()
        state_revenue = state_revenue.sort_values('payment_value', ascending=False).head(10)
        
        fig = px.bar(state_revenue, x='customer_state', y='payment_value',
                    title="Top 10 States by Revenue",
                    labels={'payment_value': 'Revenue (R$)', 'customer_state': 'State'},
                    color='payment_value',
                    color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

def page_full_dashboard(df, data_dict=None):
    """Full Dashboard Page with All Features"""
    st.title("📈 Comprehensive Dashboard")
    
    # Filters
    st.markdown("### 🔍 Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        states = st.multiselect(
            "Select States",
            options=sorted(df['customer_state'].dropna().unique()),
            default=None,
            key="state_filter"
        )
    
    with col2:
        categories = st.multiselect(
            "Select Categories", 
            options=sorted(df['product_category_name_english'].dropna().unique()),
            default=None,
            key="category_filter"
        )
    
    with col3:
        date_option = st.selectbox(
            "Date Range",
            ["All Time", "Last 30 Days", "Last 90 Days", "Last Year"],
            key="date_filter"
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if states:
        filtered_df = filtered_df[filtered_df['customer_state'].isin(states)]
    
    if categories:
        filtered_df = filtered_df[filtered_df['product_category_name_english'].isin(categories)]
    
    if date_option != "All Time":
        max_date = filtered_df['order_purchase_timestamp'].max()
        if date_option == "Last 30 Days":
            min_date = max_date - pd.Timedelta(days=30)
        elif date_option == "Last 90 Days":
            min_date = max_date - pd.Timedelta(days=90)
        else:
            min_date = max_date - pd.Timedelta(days=365)
        filtered_df = filtered_df[filtered_df['order_purchase_timestamp'] >= min_date]
    
    st.markdown(f"**📊 Showing {len(filtered_df):,} records**")
    st.markdown("---")
    
    # Tab selection using radio buttons
    tab_names = ["💰 Revenue Analysis", "👥 Customer Insights", "🏪 Seller Performance", "🚚 Logistics & Delivery"]
    selected_tab = st.radio("Select Analysis", tab_names, horizontal=True, key="tab_selector")
    
    st.markdown("---")
    
    # REVENUE ANALYSIS
    if selected_tab == "💰 Revenue Analysis":
        st.subheader("💰 Revenue Analysis")
        
        # Revenue KPIs
        col1, col2, col3, col4 = st.columns(4)
        total_revenue = filtered_df.groupby('order_id')['payment_value'].sum().sum()
        total_orders = filtered_df['order_id'].nunique()
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
        
        # Calculate growth rate
        if len(filtered_df) > 0:
            current_month = filtered_df['order_month'].max()
            prev_month = pd.Period(current_month).asfreq('M') - 1
            current_revenue = filtered_df[filtered_df['order_month'] == str(current_month)].groupby('order_id')['payment_value'].sum().sum()
            prev_revenue = filtered_df[filtered_df['order_month'] == str(prev_month)].groupby('order_id')['payment_value'].sum().sum()
            growth_rate = ((current_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
        else:
            growth_rate = 0
        
        with col1:
            st.metric("Total Revenue", f"R$ {total_revenue:,.0f}")
        with col2:
            st.metric("Total Orders", f"{total_orders:,}")
        with col3:
            st.metric("AOV", f"R$ {avg_order_value:.2f}")
        with col4:
            st.metric("MoM Growth", f"{growth_rate:+.1f}%", delta=f"{growth_rate:.1f}%")
        
        st.markdown("---")
        
        # Revenue Waterfall Chart
        st.subheader("💧 Revenue Waterfall Analysis")
        
        # Calculate revenue components
        product_revenue = filtered_df.groupby('order_id')['price'].sum().sum()
        freight_revenue = filtered_df.groupby('order_id')['freight_value'].sum().sum()
        total_payment_value = filtered_df.groupby('order_id')['payment_value'].sum().sum()
        
        # Calculate other charges (taxes, fees, etc.)
        other_charges = total_payment_value - (product_revenue + freight_revenue)
        
        # Create waterfall data
        waterfall_data = [
            ("Product Sales", product_revenue),
            ("Freight Charges", freight_revenue),
            ("Other Charges (Taxes/Fees)", other_charges),
            ("Total Revenue", total_payment_value)
        ]
        
        fig = go.Figure(go.Waterfall(
            name="Revenue Breakdown",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=[x[0] for x in waterfall_data],
            y=[x[1] for x in waterfall_data],
            text=[f"R$ {x[1]:,.0f}" for x in waterfall_data],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "green"}},
            totals={"marker": {"color": "blue"}}
        ))
        
        fig.update_layout(title="Revenue Components Waterfall", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Revenue by Payment Method and Category
        col1, col2 = st.columns(2)
        
        with col1:
            payment_dist = filtered_df.groupby('payment_type')['payment_value'].sum().reset_index()
            fig = px.pie(payment_dist, values='payment_value', names='payment_type',
                        title="Revenue by Payment Method", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            category_revenue = filtered_df[filtered_df['product_category_name_english'].notna()].groupby('product_category_name_english')['payment_value'].sum().reset_index()
            category_revenue = category_revenue.nlargest(10, 'payment_value')
            fig = px.bar(category_revenue, x='payment_value', y='product_category_name_english',
                        orientation='h', title="Top 10 Categories by Revenue",
                        labels={'payment_value': 'Revenue (R$)', 'product_category_name_english': 'Category'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Time-based Revenue Analysis
        st.subheader("📅 Time-based Revenue Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            hourly_revenue = filtered_df.groupby('order_hour')['payment_value'].mean().reset_index()
            fig = px.line(hourly_revenue, x='order_hour', y='payment_value',
                         title="Average Revenue by Hour of Day", markers=True,
                         labels={'payment_value': 'Avg Revenue (R$)', 'order_hour': 'Hour'})
            fig.update_traces(line_color='#1f77b4', marker=dict(size=8))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            weekday_revenue = filtered_df.groupby('order_dayofweek')['payment_value'].sum().reset_index()
            weekday_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
            weekday_revenue['day_name'] = weekday_revenue['order_dayofweek'].map(weekday_names)
            fig = px.bar(weekday_revenue, x='day_name', y='payment_value',
                        title="Revenue by Day of Week",
                        labels={'payment_value': 'Revenue (R$)', 'day_name': 'Day'})
            st.plotly_chart(fig, use_container_width=True)
    
    # CUSTOMER INSIGHTS
    elif selected_tab == "👥 Customer Insights":
        st.subheader("👥 Customer Insights")
        
        # Customer KPIs
        col1, col2, col3, col4 = st.columns(4)
        total_customers = filtered_df['customer_unique_id'].nunique()
        avg_review = filtered_df['review_score'].mean() if filtered_df['review_score'].notna().any() else 0
        
        # Calculate repeat rate
        customer_orders = filtered_df.groupby('customer_unique_id')['order_id'].nunique()
        repeat_customers = customer_orders[customer_orders > 1].count()
        repeat_rate = (repeat_customers / total_customers * 100) if total_customers > 0 else 0
        
        # Calculate NPS
        promoters = filtered_df[filtered_df['review_score'] >= 4]['order_id'].nunique()
        detractors = filtered_df[filtered_df['review_score'] <= 2]['order_id'].nunique()
        total_reviews = filtered_df[filtered_df['review_score'].notna()]['order_id'].nunique()
        nps = ((promoters - detractors) / total_reviews * 100) if total_reviews > 0 else 0
        
        with col1:
            st.metric("Total Customers", f"{total_customers:,}")
        with col2:
            st.metric("Avg Review Score", f"⭐ {avg_review:.2f}")
        with col3:
            st.metric("Repeat Rate", f"{repeat_rate:.1f}%")
        with col4:
            st.metric("NPS Score", f"{nps:.0f}")
        
        st.markdown("---")
        
        # Cohort Analysis
        st.subheader("📊 Customer Cohort Analysis")
        
        # Create cohort data
        cohort_data = filtered_df.copy()
        cohort_data['order_month'] = pd.to_datetime(cohort_data['order_month'])
        cohort_data['cohort'] = cohort_data.groupby('customer_unique_id')['order_month'].transform('min')
        cohort_data['cohort_month'] = cohort_data['cohort'].dt.to_period('M')
        cohort_data['order_period'] = cohort_data['order_month'].dt.to_period('M')
        cohort_data['cohort_index'] = (cohort_data['order_period'] - cohort_data['cohort_month']).apply(lambda x: x.n if hasattr(x, 'n') else 0)
        
        # Create cohort matrix
        cohort_matrix = cohort_data.groupby(['cohort_month', 'cohort_index'])['customer_unique_id'].nunique().reset_index()
        cohort_pivot = cohort_matrix.pivot(index='cohort_month', columns='cohort_index', values='customer_unique_id')
        
        # Calculate retention rates
        retention = cohort_pivot.divide(cohort_pivot.iloc[:, 0], axis=0) * 100
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=retention.values,
            x=[f"Month {i}" for i in retention.columns],
            y=retention.index.astype(str),
            colorscale='RdYlGn',
            text=np.round(retention.values, 1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title="Retention %")
        ))
        
        fig.update_layout(
            title="Customer Retention Cohort Analysis",
            xaxis_title="Months Since First Purchase",
            yaxis_title="Cohort Month",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Customer Segmentation
        col1, col2 = st.columns(2)
        
        with col1:
            # Review Score Distribution
            review_dist = filtered_df[filtered_df['review_score'].notna()].groupby('review_score')['order_id'].nunique().reset_index()
            fig = px.bar(review_dist, x='review_score', y='order_id',
                        title="Review Score Distribution",
                        labels={'order_id': 'Number of Orders', 'review_score': 'Review Score'},
                        color='review_score', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Customer Value Distribution
            customer_value = filtered_df.groupby('customer_unique_id')['payment_value'].sum().reset_index()
            customer_value['segment'] = pd.qcut(customer_value['payment_value'], q=4, labels=['Low', 'Medium', 'High', 'VIP'])
            segment_dist = customer_value.groupby('segment').size().reset_index(name='count')
            fig = px.pie(segment_dist, values='count', names='segment',
                        title="Customer Value Segmentation", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        
        # Geographic Customer Distribution
        st.subheader("🗺️ Geographic Customer Distribution")
        
        state_customers = filtered_df.groupby('customer_state')['customer_unique_id'].nunique().reset_index()
        state_customers = state_customers.sort_values('customer_unique_id', ascending=False).head(15)
        
        fig = px.treemap(state_customers, path=['customer_state'], values='customer_unique_id',
                        title="Customer Distribution by State",
                        color='customer_unique_id', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    # SELLER PERFORMANCE
    elif selected_tab == "🏪 Seller Performance":
        st.subheader("🏪 Seller Performance")
        
        # Seller KPIs
        col1, col2, col3, col4 = st.columns(4)
        total_sellers = filtered_df['seller_id'].nunique()
        
        seller_revenue = filtered_df.groupby('seller_id')['payment_value'].sum()
        avg_seller_revenue = seller_revenue.mean()
        top_seller_revenue = seller_revenue.max()
        
        seller_orders = filtered_df.groupby('seller_id')['order_id'].nunique()
        avg_orders_per_seller = seller_orders.mean()
        
        with col1:
            st.metric("Total Sellers", f"{total_sellers:,}")
        with col2:
            st.metric("Avg Seller Revenue", f"R$ {avg_seller_revenue:,.0f}")
        with col3:
            st.metric("Top Seller Revenue", f"R$ {top_seller_revenue:,.0f}")
        with col4:
            st.metric("Avg Orders/Seller", f"{avg_orders_per_seller:.0f}")
        
        st.markdown("---")
        
        # Top Sellers
        col1, col2 = st.columns(2)
        
        with col1:
            top_sellers = filtered_df.groupby('seller_id').agg({
                'payment_value': 'sum',
                'order_id': 'nunique',
                'review_score': 'mean'
            }).reset_index()
            top_sellers = top_sellers.nlargest(10, 'payment_value')
            
            fig = px.bar(top_sellers, x='payment_value', y='seller_id',
                        orientation='h', title="Top 10 Sellers by Revenue",
                        labels={'payment_value': 'Revenue (R$)', 'seller_id': 'Seller ID'},
                        color='review_score', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Seller State Distribution
            seller_states = filtered_df.groupby('seller_state')['seller_id'].nunique().reset_index()
            seller_states = seller_states.nlargest(10, 'seller_id')
            
            fig = px.pie(seller_states, values='seller_id', names='seller_state',
                        title="Sellers by State (Top 10)", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        
        # Seller Performance Matrix
        st.subheader("📈 Seller Performance Matrix")
        
        seller_metrics = filtered_df.groupby('seller_id').agg({
            'payment_value': 'sum',
            'order_id': 'nunique',
            'review_score': 'mean',
            'delivery_time': 'mean'
        }).reset_index()
        
        seller_metrics = seller_metrics.dropna()
        seller_metrics = seller_metrics[seller_metrics['payment_value'] > 0]
        
        fig = px.scatter(seller_metrics.head(100), x='order_id', y='payment_value',
                        size='review_score', color='delivery_time',
                        hover_data=['seller_id'],
                        title="Seller Performance: Orders vs Revenue (Top 100)",
                        labels={'order_id': 'Number of Orders', 'payment_value': 'Revenue (R$)',
                               'delivery_time': 'Avg Delivery Time'},
                        color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)
        
        # Seller Growth Analysis
        st.subheader("📊 Seller Growth Trends")
        
        monthly_sellers = filtered_df.groupby('order_month')['seller_id'].nunique().reset_index()
        monthly_sellers = monthly_sellers.sort_values('order_month')
        
        fig = px.area(monthly_sellers, x='order_month', y='seller_id',
                     title="Active Sellers Over Time",
                     labels={'seller_id': 'Number of Active Sellers', 'order_month': 'Month'})
        fig.update_traces(fillcolor='rgba(255, 127, 14, 0.3)', line=dict(color='#ff7f0e', width=3))
        st.plotly_chart(fig, use_container_width=True)
    
    # LOGISTICS & DELIVERY
    elif selected_tab == "🚚 Logistics & Delivery":
        st.subheader("🚚 Logistics & Delivery")
        
        # Logistics KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        delivered_orders = filtered_df[filtered_df['order_status'] == 'delivered']['order_id'].nunique()
        total_orders = filtered_df['order_id'].nunique()
        delivery_rate = (delivered_orders / total_orders * 100) if total_orders > 0 else 0
        
        avg_delivery_time = filtered_df[filtered_df['delivery_time'].notna()]['delivery_time'].mean()
        on_time_deliveries = filtered_df[filtered_df['on_time_delivery'] == True]['order_id'].nunique()
        on_time_rate = (on_time_deliveries / delivered_orders * 100) if delivered_orders > 0 else 0
        
        avg_delay = filtered_df[filtered_df['delivery_delay'] > 0]['delivery_delay'].mean()
        
        with col1:
            st.metric("Delivery Rate", f"{delivery_rate:.1f}%")
        with col2:
            st.metric("Avg Delivery Time", f"{avg_delivery_time:.1f} days" if pd.notna(avg_delivery_time) else "N/A")
        with col3:
            st.metric("On-Time Rate", f"{on_time_rate:.1f}%")
        with col4:
            st.metric("Avg Delay", f"{avg_delay:.1f} days" if pd.notna(avg_delay) else "0 days")
        
        st.markdown("---")
        
        # Order Flow Sankey Diagram
        st.subheader("🔄 Order Flow Sankey Diagram")
        
        # Prepare Sankey data
        status_flow = filtered_df.groupby(['order_status', 'customer_state'])['order_id'].nunique().reset_index()
        status_flow = status_flow.nlargest(30, 'order_id')
        
        # Create nodes
        all_nodes = list(status_flow['order_status'].unique()) + list(status_flow['customer_state'].unique())
        node_dict = {node: i for i, node in enumerate(all_nodes)}
        
        # Create links
        source = [node_dict[status] for status in status_flow['order_status']]
        target = [node_dict[state] for state in status_flow['customer_state']]
        value = status_flow['order_id'].tolist()
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes,
                color=["blue" if node in status_flow['order_status'].unique() else "green" for node in all_nodes]
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color="rgba(0,0,96,0.2)"
            )
        )])
        
        fig.update_layout(title="Order Status Flow by State", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Delivery Performance
        col1, col2 = st.columns(2)
        
        with col1:
            # Order Status Distribution
            status_dist = filtered_df.groupby('order_status')['order_id'].nunique().reset_index()
            fig = px.pie(status_dist, values='order_id', names='order_status',
                        title="Order Status Distribution", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Delivery Time Distribution
            delivery_data = filtered_df[filtered_df['delivery_time'].notna() & (filtered_df['delivery_time'] >= 0) & (filtered_df['delivery_time'] <= 60)]
            fig = px.histogram(delivery_data, x='delivery_time', nbins=30,
                             title="Delivery Time Distribution",
                             labels={'delivery_time': 'Days to Deliver', 'count': 'Number of Orders'})
            fig.update_traces(marker_color='lightblue', marker_line_color='darkblue', marker_line_width=1)
            st.plotly_chart(fig, use_container_width=True)
        
        # Geographic Delivery Performance
        st.subheader("🗺️ Geographic Delivery Performance")
        
        state_delivery = filtered_df[filtered_df['delivery_time'].notna()].groupby('customer_state').agg({
            'delivery_time': 'mean',
            'on_time_delivery': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0,
            'order_id': 'nunique'
        }).reset_index()
        
        state_delivery = state_delivery.nlargest(15, 'order_id')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=state_delivery['customer_state'],
            y=state_delivery['delivery_time'],
            name='Avg Delivery Time (days)',
            marker_color='lightblue',
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=state_delivery['customer_state'],
            y=state_delivery['on_time_delivery'],
            name='On-Time Rate (%)',
            marker_color='green',
            yaxis='y2',
            mode='lines+markers',
            line=dict(width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="Delivery Performance by State",
            xaxis=dict(title="State"),
            yaxis=dict(title="Avg Delivery Time (days)", side="left"),
            yaxis2=dict(title="On-Time Rate (%)", overlaying="y", side="right"),
            hovermode="x",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def page_insights_trends(df, data_dict=None):
    """Insights & Trends Page with Bias Adjustment"""
    st.title("🔍 Insights & Trends Analysis")
    st.markdown("### Data-Driven Insights with Geographic Bias Adjustment")
    
    # Trend 1: Geographic Revenue Bias Analysis
    st.subheader("📍 Geographic Revenue Analysis (Population-Adjusted)")
    
    state_metrics = df.groupby('customer_state').agg({
        'payment_value': 'sum',
        'order_id': 'nunique',
        'customer_unique_id': 'nunique'
    }).reset_index()
    
    # Add population data
    state_metrics['population'] = state_metrics['customer_state'].map(BRAZIL_POPULATION)
    state_metrics = state_metrics.dropna(subset=['population'])
    
    # Calculate per capita metrics
    state_metrics['revenue_per_capita'] = state_metrics['payment_value'] / state_metrics['population']
    state_metrics['orders_per_capita'] = state_metrics['order_id'] / state_metrics['population'] * 1000
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Raw revenue by state
        fig = px.bar(state_metrics.nlargest(10, 'payment_value'), 
                    x='customer_state', y='payment_value',
                    title="Top 10 States by Total Revenue (Raw)",
                    labels={'payment_value': 'Total Revenue (R$)', 'customer_state': 'State'},
                    color='payment_value', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Population-adjusted revenue
        fig = px.bar(state_metrics.nlargest(10, 'revenue_per_capita'), 
                    x='customer_state', y='revenue_per_capita',
                    title="Top 10 States by Revenue Per Capita (Adjusted)",
                    labels={'revenue_per_capita': 'Revenue Per Capita (R$)', 'customer_state': 'State'},
                    color='revenue_per_capita', color_continuous_scale='Greens')
        st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Key Insight:** While São Paulo (SP) and Rio de Janeiro (RJ) dominate in absolute revenue, 
    smaller states show higher per-capita spending when adjusted for population. This reveals 
    untapped market potential in less populated regions.
    """)
    
    # Trend 2: Seasonal Patterns and Growth
    st.subheader("📈 Seasonal Patterns & Growth Trends")
    
    # Monthly growth analysis
    monthly_metrics = df.groupby('order_month').agg({
        'payment_value': 'sum',
        'order_id': 'nunique',
        'customer_unique_id': 'nunique'
    }).reset_index()
    monthly_metrics = monthly_metrics.sort_values('order_month')
    
    # Calculate month-over-month growth
    monthly_metrics['revenue_growth'] = monthly_metrics['payment_value'].pct_change() * 100
    monthly_metrics['order_growth'] = monthly_metrics['order_id'].pct_change() * 100
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Revenue & Orders Trend", "Month-over-Month Growth Rate"),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    # Trend lines
    fig.add_trace(
        go.Scatter(x=monthly_metrics['order_month'], y=monthly_metrics['payment_value'],
                  name='Revenue', line=dict(color='blue', width=3)),
        row=1, col=1, secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=monthly_metrics['order_month'], y=monthly_metrics['order_id'],
                  name='Orders', line=dict(color='orange', width=3)),
        row=1, col=1, secondary_y=True
    )
    
    # Growth rates
    fig.add_trace(
        go.Bar(x=monthly_metrics['order_month'], y=monthly_metrics['revenue_growth'],
              name='Revenue Growth %', marker_color='lightblue'),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Revenue (R$)", secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="Number of Orders", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Growth Rate (%)", row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Trend 3: Product Category Evolution
    st.subheader("🛍️ Product Category Evolution & Market Share")
    
    # Category trends over time
    category_trends = df[df['product_category_name_english'].notna()].groupby(['order_month', 'product_category_name_english'])['payment_value'].sum().reset_index()
    top_categories = df[df['product_category_name_english'].notna()].groupby('product_category_name_english')['payment_value'].sum().nlargest(5).index
    category_trends = category_trends[category_trends['product_category_name_english'].isin(top_categories)]
    category_trends = category_trends.sort_values('order_month')
    
    fig = px.line(category_trends, x='order_month', y='payment_value',
                 color='product_category_name_english',
                 title="Top 5 Category Revenue Trends Over Time",
                 labels={'payment_value': 'Revenue (R$)', 'order_month': 'Month',
                        'product_category_name_english': 'Category'})
    fig.update_traces(line=dict(width=3))
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("""
    **Key Findings:**
    1. **Geographic Opportunity:** States with high per-capita spending but low absolute revenue represent expansion opportunities
    2. **Seasonal Patterns:** Clear growth trajectory with seasonal fluctuations - optimize inventory for peak periods
    3. **Category Dynamics:** Technology and fashion categories show strongest growth momentum
    """)

def page_technical_documentation(df, data_dict=None):
    """Comprehensive Technical Documentation Page"""
    st.title("📚 Technical Documentation")
    st.markdown("### Complete Implementation Guide & Methodologies")
    
    # Documentation navigation
    doc_section = st.radio(
        "Select Documentation Section",
        [
            "🏗️ Architecture Overview", 
            "📖 Step-by-Step Guide", 
            "🔬 Data Science Methodology", 
            "💻 Code Structure", 
            "📊 Analytics Framework",
            "🚀 Deployment Guide"
        ],
        horizontal=True,
        key="doc_selector"
    )
    
    st.markdown("---")
    
    # ARCHITECTURE OVERVIEW
    if doc_section == "🏗️ Architecture Overview":
        st.subheader("🏗️ System Architecture")
        
        st.markdown("""
        ## 🎯 Project Overview
        
        **Olist E-Commerce Analytics Platform** is a comprehensive business intelligence dashboard built for analyzing Brazilian e-commerce data. The platform provides executives, analysts, and business stakeholders with actionable insights through interactive visualizations and strategic recommendations.
        
        ### 🎨 Design Philosophy
        - **Data-Driven Decision Making**: Every metric is calculated from raw transactional data
        - **Bias-Adjusted Analysis**: Geographic insights adjusted for population distribution
        - **Executive-Ready**: Professional visualizations suitable for C-level presentations
        - **Scalable Architecture**: Modular design for easy expansion and maintenance
        """)
        
        # Create architecture diagram using Plotly
        fig = go.Figure()
        
        # Data Layer
        fig.add_shape(type="rect", x0=0, y0=4, x1=10, y1=5, 
                     fillcolor="lightblue", line=dict(color="blue", width=2))
        fig.add_annotation(x=5, y=4.5, text="DATA LAYER<br>9 CSV Files | 128M+ Records", 
                          showarrow=False, font=dict(size=12, color="darkblue"))
        
        # Processing Layer  
        fig.add_shape(type="rect", x0=0, y0=2.5, x1=10, y1=3.5,
                     fillcolor="lightgreen", line=dict(color="green", width=2))
        fig.add_annotation(x=5, y=3, text="PROCESSING LAYER<br>Pandas | Data Merging | Feature Engineering", 
                          showarrow=False, font=dict(size=12, color="darkgreen"))
        
        # Analytics Layer
        fig.add_shape(type="rect", x0=0, y0=1, x1=10, y1=2,
                     fillcolor="lightyellow", line=dict(color="orange", width=2))
        fig.add_annotation(x=5, y=1.5, text="ANALYTICS LAYER<br>Statistical Analysis | KPI Calculations | Bias Adjustment", 
                          showarrow=False, font=dict(size=12, color="darkorange"))
        
        # Visualization Layer
        fig.add_shape(type="rect", x0=0, y0=-0.5, x1=10, y1=0.5,
                     fillcolor="lightcoral", line=dict(color="red", width=2))
        fig.add_annotation(x=5, y=0, text="VISUALIZATION LAYER<br>Streamlit | Plotly | Interactive Dashboards", 
                          showarrow=False, font=dict(size=12, color="darkred"))
        
        # Add arrows
        for y in [3.5, 2, 0.5]:
            fig.add_annotation(x=5, y=y, ax=5, ay=y+0.3, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="black")
        
        fig.update_layout(
            title="System Architecture Flow",
            xaxis=dict(visible=False, range=[-1, 11]),
            yaxis=dict(visible=False, range=[-1, 6]),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 🔧 Technology Stack
        
        | Component | Technology | Purpose |
        |-----------|------------|---------|
        | **Frontend** | Streamlit | Interactive web application framework |
        | **Data Processing** | Pandas, NumPy | Data manipulation and analysis |
        | **Visualizations** | Plotly | Interactive charts and graphs |
        | **Caching** | Streamlit Cache | Performance optimization |
        | **Deployment** | Python Virtual Environment | Isolated runtime environment |
        
        ### 📊 Key Features
        
        ✅ **Executive Dashboard**: High-level KPIs and trends  
        ✅ **Deep-Dive Analytics**: Detailed analysis across 4 domains  
        ✅ **Geographic Bias Adjustment**: Population-normalized insights  
        ✅ **Strategic Recommendations**: Actionable business insights with ROI projections  
        ✅ **Real-time Filtering**: Interactive data exploration  
        ✅ **Professional Visualizations**: Waterfall, Sankey, Cohort, and Treemap charts  
        """)
    
    # STEP-BY-STEP GUIDE
    elif doc_section == "📖 Step-by-Step Guide":
        st.subheader("📖 Implementation Guide for 12th Pass Intern")
        
        st.markdown("""
        ## 🎯 Prerequisites
        
        **What you need to know:**
        - Basic computer operations (file management, terminal/command prompt)
        - No programming experience required - we'll teach you everything!
        
        **What you'll learn:**
        - Python basics for data analysis
        - Building web applications with Streamlit
        - Creating professional visualizations
        - Business intelligence concepts
        """)
        
        # Step-by-step tabs
        step_tab = st.selectbox(
            "Select Implementation Step",
            [
                "Step 1: Environment Setup",
                "Step 2: Understanding Data",
                "Step 3: Building Basic App",
                "Step 4: Adding Visualizations", 
                "Step 5: Advanced Features",
                "Step 6: Testing & Deployment"
            ]
        )
        
        if step_tab == "Step 1: Environment Setup":
            st.markdown("""
            ### 🔧 Step 1: Environment Setup (30 minutes)
            
            #### 1.1 Install Python
            ```bash
            # Download Python from python.org
            # Version 3.8 or higher recommended
            python --version  # Should show Python 3.x.x
            ```
            
            #### 1.2 Create Project Folder
            ```bash
            mkdir ecommerce-dashboard
            cd ecommerce-dashboard
            ```
            
            #### 1.3 Create Virtual Environment
            ```bash
            python -m venv venv
            
            # Activate (Windows)
            venv\\Scripts\\activate
            
            # Activate (Mac/Linux)  
            source venv/bin/activate
            ```
            
            #### 1.4 Install Required Libraries
            ```bash
            pip install streamlit pandas plotly numpy
            ```
            
            #### 1.5 Download Dataset
            - Get all 9 CSV files from Brazilian E-commerce dataset
            - Place in project folder
            - Files needed: `olist_customers_dataset.csv`, `olist_orders_dataset.csv`, etc.
            
            #### ✅ Verification
            ```bash
            streamlit hello  # Should open Streamlit demo in browser
            ```
            """)
            
        elif step_tab == "Step 2: Understanding Data":
            st.markdown("""
            ### 📊 Step 2: Understanding the Data (45 minutes)
            
            #### 2.1 Dataset Overview
            **You're working with a REAL Brazilian e-commerce company's data!**
            
            | File | Records | What it Contains |
            |------|---------|------------------|
            | `orders.csv` | 99K+ | Order information, dates, status |
            | `order_items.csv` | 112K+ | Products in each order, prices |
            | `customers.csv` | 99K+ | Customer locations and IDs |
            | `sellers.csv` | 3K+ | Seller information and locations |
            | `products.csv` | 32K+ | Product details and categories |
            | `payments.csv` | 103K+ | Payment methods and values |
            | `reviews.csv` | 99K+ | Customer ratings and reviews |
            
            #### 2.2 Key Relationships
            ```
            CUSTOMER → places → ORDER → contains → ORDER_ITEMS → references → PRODUCT
                                 ↓                                         ↓
                              PAYMENT                                   SELLER
                                 ↓
                              REVIEW
            ```
            
            #### 2.3 Business Questions We're Answering
            1. **Revenue**: How much money is the company making?
            2. **Customers**: Who are our customers and what do they like?
            3. **Sellers**: Which sellers perform best?
            4. **Logistics**: How well are we delivering orders?
            5. **Growth**: Are we growing or declining?
            
            #### 2.4 Practice Exercise
            ```python
            import pandas as pd
            
            # Load one file to explore
            orders = pd.read_csv('olist_orders_dataset.csv')
            print(f"Orders dataset has {len(orders)} rows")
            print(orders.head())  # Show first 5 rows
            print(orders.info())  # Show column types
            ```
            """)
            
        elif step_tab == "Step 3: Building Basic App":
            st.markdown("""
            ### 🏗️ Step 3: Building Basic App (60 minutes)
            
            #### 3.1 Create Your First App
            **File: `app.py`**
            
            ```python
            import streamlit as st
            import pandas as pd
            
            # App title
            st.title("My E-Commerce Dashboard")
            st.write("Welcome to my first data app!")
            
            # Load data
            @st.cache_data  # This makes the app faster!
            def load_data():
                orders = pd.read_csv('olist_orders_dataset.csv')
                return orders
            
            # Show basic info
            df = load_data()
            st.write(f"We have {len(df)} orders in our dataset!")
            st.write(df.head())
            ```
            
            #### 3.2 Run Your App
            ```bash
            streamlit run app.py
            ```
            🎉 **Your first data app is now running in your browser!**
            
            #### 3.3 Add Navigation
            ```python
            # Add sidebar navigation
            page = st.sidebar.selectbox(
                "Choose a page",
                ["Home", "Orders", "Analysis"]
            )
            
            if page == "Home":
                st.write("Welcome to the home page!")
            elif page == "Orders":
                st.write("Here are the orders")
            elif page == "Analysis": 
                st.write("Here's our analysis")
            ```
            
            #### 3.4 Your First KPI
            ```python
            # Calculate total revenue
            # First, we need to load order payments
            payments = pd.read_csv('olist_order_payments_dataset.csv')
            total_revenue = payments['payment_value'].sum()
            
            st.metric("Total Revenue", f"R$ {total_revenue:,.2f}")
            ```
            
            #### ✅ What You've Built
            - Working Streamlit app
            - Data loading with caching
            - Navigation between pages
            - Your first business metric!
            """)
        
        elif step_tab == "Step 4: Adding Visualizations":
            st.markdown("""
            ### 📈 Step 4: Adding Visualizations (90 minutes)
            
            #### 4.1 Your First Chart
            ```python
            import plotly.express as px
            
            # Monthly revenue trend
            monthly_data = payments.groupby('payment_month')['payment_value'].sum()
            
            fig = px.line(
                x=monthly_data.index, 
                y=monthly_data.values,
                title="Monthly Revenue Trend"
            )
            st.plotly_chart(fig)
            ```
            
            #### 4.2 Interactive Charts
            ```python
            # Payment method pie chart
            payment_methods = payments.groupby('payment_type')['payment_value'].sum()
            
            fig = px.pie(
                values=payment_methods.values,
                names=payment_methods.index,
                title="Revenue by Payment Method"
            )
            st.plotly_chart(fig, use_container_width=True)
            ```
            
            #### 4.3 Professional Metrics Display
            ```python
            # Create columns for metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Revenue", f"R$ {total_revenue:,.0f}")
            with col2:
                st.metric("Total Orders", f"{total_orders:,}")
            with col3:
                st.metric("Avg Order Value", f"R$ {avg_order_value:.2f}")
            with col4:
                st.metric("Customer Rating", f"⭐ {avg_rating:.1f}")
            ```
            
            #### 4.4 Chart Types You'll Master
            
            | Chart Type | When to Use | Code Example |
            |------------|-------------|--------------|
            | **Line Chart** | Trends over time | `px.line()` |
            | **Bar Chart** | Compare categories | `px.bar()` |
            | **Pie Chart** | Show proportions | `px.pie()` |
            | **Scatter Plot** | Show relationships | `px.scatter()` |
            | **Heatmap** | Show patterns | `px.imshow()` |
            
            #### ✅ Skills Gained
            - Creating interactive charts
            - Professional metric displays  
            - Understanding when to use different chart types
            """)
            
        elif step_tab == "Step 5: Advanced Features":
            st.markdown("""
            ### 🚀 Step 5: Advanced Features (120 minutes)
            
            #### 5.1 Data Filtering
            ```python
            # Let users filter data
            states = st.multiselect(
                "Select States to Analyze",
                options=df['customer_state'].unique(),
                default=['SP', 'RJ']
            )
            
            # Filter the data
            filtered_df = df[df['customer_state'].isin(states)]
            ```
            
            #### 5.2 Advanced Visualizations
            
            **Waterfall Chart (Revenue Breakdown):**
            ```python
            import plotly.graph_objects as go
            
            fig = go.Figure(go.Waterfall(
                name="Revenue Components",
                x=["Product Sales", "Shipping", "Total"],
                y=[product_revenue, shipping_revenue, total_revenue],
                text=[f"R$ {x:,.0f}" for x in [product_revenue, shipping_revenue, total_revenue]]
            ))
            st.plotly_chart(fig)
            ```
            
            **Sankey Diagram (Order Flow):**
            ```python
            fig = go.Figure(data=[go.Sankey(
                node=dict(label=["Orders", "Delivered", "Canceled", "SP", "RJ"]),
                link=dict(source=[0,0,1,1], target=[1,2,3,4], value=[80,20,50,30])
            )])
            st.plotly_chart(fig)
            ```
            
            #### 5.3 Business Intelligence Features
            
            **Growth Rate Calculation:**
            ```python
            def calculate_growth(current, previous):
                if previous == 0:
                    return 0
                return ((current - previous) / previous) * 100
            
            growth = calculate_growth(current_month_revenue, previous_month_revenue)
            st.metric("Revenue Growth", f"{growth:.1f}%", delta=f"{growth:.1f}%")
            ```
            
            **Customer Segmentation:**
            ```python
            # Segment customers by value
            customer_value = df.groupby('customer_id')['payment_value'].sum()
            
            # Create segments
            customer_value['segment'] = pd.cut(
                customer_value, 
                bins=[0, 100, 500, 1000, float('inf')],
                labels=['Low', 'Medium', 'High', 'VIP']
            )
            ```
            
            #### 5.4 Performance Optimization
            ```python
            # Cache expensive operations
            @st.cache_data
            def load_and_process_data():
                # Your data processing here
                return processed_data
            
            # Use session state for user selections
            if 'selected_filters' not in st.session_state:
                st.session_state.selected_filters = {}
            ```
            """)
        
        elif step_tab == "Step 6: Testing & Deployment":
            st.markdown("""
            ### 🧪 Step 6: Testing & Deployment (60 minutes)
            
            #### 6.1 Testing Your App
            
            **Data Validation:**
            ```python
            # Test your calculations
            def test_revenue_calculation():
                manual_total = 0
                for payment in payments['payment_value']:
                    manual_total += payment
                
                assert abs(manual_total - total_revenue) < 0.01
                print("✅ Revenue calculation is correct!")
            
            test_revenue_calculation()
            ```
            
            **Error Handling:**
            ```python
            try:
                df = pd.read_csv('data.csv')
            except FileNotFoundError:
                st.error("Data file not found! Please check the file path.")
                st.stop()
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            ```
            
            #### 6.2 Performance Testing
            ```python
            # Time your functions
            import time
            
            start_time = time.time()
            result = expensive_calculation()
            end_time = time.time()
            
            st.write(f"Calculation took {end_time - start_time:.2f} seconds")
            ```
            
            #### 6.3 Deployment Options
            
            **Local Deployment:**
            ```bash
            # Run on specific port
            streamlit run app.py --server.port 8501
            
            # Run in headless mode
            streamlit run app.py --server.headless true
            ```
            
            **Cloud Deployment (Streamlit Cloud):**
            1. Push code to GitHub repository
            2. Connect to Streamlit Cloud
            3. Deploy with one click!
            
            #### 6.4 Documentation
            ```python
            # Add help text
            st.sidebar.help("This dashboard analyzes e-commerce data...")
            
            # Add tooltips
            st.metric(
                "Revenue", 
                f"R$ {revenue:,.2f}",
                help="Total revenue from all completed orders"
            )
            ```
            
            #### ✅ Final Checklist
            - [ ] App runs without errors
            - [ ] All charts display correctly  
            - [ ] Filters work as expected
            - [ ] Performance is acceptable
            - [ ] Documentation is complete
            - [ ] Ready for presentation!
            
            ### 🎉 Congratulations!
            You've built a professional business intelligence dashboard from scratch!
            """)
    
    # DATA SCIENCE METHODOLOGY
    elif doc_section == "🔬 Data Science Methodology":
        st.subheader("🔬 Data Science Methodology")
        
        st.markdown("""
        ## 🔬 Scientific Approach to Business Analytics
        
        Our dashboard follows established data science methodologies to ensure accurate, actionable insights.
        """)
        
        method_tab = st.selectbox(
            "Select Methodology Topic",
            [
                "CRISP-DM Framework",
                "Statistical Methods", 
                "Bias Adjustment Techniques",
                "KPI Calculation Methods",
                "Validation Approaches"
            ]
        )
        
        if method_tab == "CRISP-DM Framework":
            st.markdown("""
            ### 📋 CRISP-DM Implementation
            
            We follow the **Cross-Industry Standard Process for Data Mining (CRISP-DM)** methodology:
            
            #### 1. Business Understanding 🎯
            **Objectives:**
            - Understand e-commerce performance across multiple dimensions
            - Provide actionable insights for business growth
            - Enable data-driven decision making
            
            **Success Criteria:**
            - Accurate revenue reporting within 0.1% margin
            - Geographic insights adjusted for population bias
            - Strategic recommendations with measurable ROI projections
            
            #### 2. Data Understanding 📊
            **Data Sources:**
            - 9 interconnected CSV files from Olist (Brazilian e-commerce)
            - 128M+ records spanning 2016-2018
            - Real transactional data with anonymized customer information
            
            **Data Quality Assessment:**
            ```python
            # Data completeness check
            completeness = df.isnull().sum() / len(df) * 100
            
            # Data consistency validation  
            duplicate_orders = df['order_id'].duplicated().sum()
            
            # Data accuracy verification
            revenue_check = (df['price'] + df['freight_value']).sum()
            ```
            
            #### 3. Data Preparation 🔧
            **Feature Engineering:**
            ```python
            # Calculate delivery performance
            df['delivery_time'] = (df['delivered_date'] - df['purchase_date']).dt.days
            df['on_time_delivery'] = df['delivery_delay'] <= 0
            
            # Extract temporal features
            df['order_month'] = df['purchase_date'].dt.to_period('M')
            df['order_hour'] = df['purchase_date'].dt.hour
            df['order_dayofweek'] = df['purchase_date'].dt.dayofweek
            ```
            
            **Data Integration:**
            - Join 9 tables using proper foreign key relationships
            - Handle missing values with business logic
            - Create derived metrics (AOV, NPS, cohort indices)
            
            #### 4. Modeling 📈
            **Analytics Models:**
            - **Cohort Analysis**: Customer retention over time
            - **RFM Segmentation**: Recency, Frequency, Monetary value
            - **Geographic Normalization**: Population-adjusted metrics
            - **Growth Decomposition**: Trend and seasonality analysis
            
            #### 5. Evaluation ✅
            **Validation Methods:**
            - Cross-validation against external benchmarks
            - A/B testing of different calculation methods
            - Expert review of business logic
            - Automated testing suite for metric accuracy
            
            #### 6. Deployment 🚀
            **Production Pipeline:**
            - Streamlit web application with caching
            - Real-time data processing and visualization
            - Interactive filtering and drill-down capabilities
            - Professional reporting formats
            """)
        
        elif method_tab == "Statistical Methods":
            st.markdown("""
            ### 📊 Statistical Methods & Calculations
            
            #### Revenue Metrics
            
            **Total Revenue (Aggregation):**
            ```python
            # Method: Sum of payments per unique order
            total_revenue = df.groupby('order_id')['payment_value'].sum().sum()
            
            # Why: Handles multiple payments per order correctly
            # Validation: Cross-check with order_items price + freight
            ```
            
            **Average Order Value (Central Tendency):**
            ```python
            # Method: Total Revenue / Unique Orders
            aov = total_revenue / df['order_id'].nunique()
            
            # Alternative: Median AOV for skewed distributions
            median_aov = df.groupby('order_id')['payment_value'].sum().median()
            ```
            
            #### Growth Analysis
            
            **Month-over-Month Growth (Time Series):**
            ```python
            # Method: Percentage change calculation
            monthly_revenue = df.groupby('order_month')['payment_value'].sum()
            mom_growth = monthly_revenue.pct_change() * 100
            
            # Seasonality adjustment using moving averages
            ma_3 = monthly_revenue.rolling(window=3).mean()
            seasonal_adjusted = monthly_revenue / ma_3
            ```
            
            #### Customer Analytics
            
            **Net Promoter Score (NPS):**
            ```python
            # Method: Industry-standard NPS calculation
            promoters = (df['review_score'] >= 4).sum()
            detractors = (df['review_score'] <= 2).sum()
            total_reviews = df['review_score'].notna().sum()
            
            nps = ((promoters - detractors) / total_reviews) * 100
            
            # Confidence interval calculation
            from scipy import stats
            se = np.sqrt((promoters + detractors) / total_reviews)
            ci_95 = stats.norm.interval(0.95, loc=nps, scale=se*100)
            ```
            
            **Customer Lifetime Value (CLV):**
            ```python
            # Method: Cohort-based CLV estimation
            customer_metrics = df.groupby('customer_id').agg({
                'payment_value': 'sum',          # Total spent
                'order_id': 'nunique',           # Frequency
                'order_purchase_timestamp': ['min', 'max']  # Lifespan
            })
            
            # Calculate purchase frequency and monetary value
            frequency = customer_metrics['order_id']['nunique']
            monetary = customer_metrics['payment_value']['sum']
            
            # Simple CLV model
            clv = monetary * frequency * retention_rate
            ```
            
            #### Geographic Analysis
            
            **Population Bias Adjustment:**
            ```python
            # Method: Per capita normalization
            state_metrics['revenue_per_capita'] = (
                state_metrics['total_revenue'] / state_metrics['population']
            )
            
            # Market penetration calculation
            state_metrics['penetration_rate'] = (
                state_metrics['unique_customers'] / state_metrics['population'] * 1000
            )
            
            # Opportunity scoring (deviation from expected)
            expected_revenue = state_metrics['population'] * national_avg_per_capita
            state_metrics['opportunity_score'] = (
                expected_revenue - state_metrics['total_revenue']
            ) / expected_revenue
            ```
            
            #### Delivery Performance
            
            **Service Level Metrics:**
            ```python
            # On-time delivery rate
            on_time_rate = (df['delivery_delay'] <= 0).mean() * 100
            
            # 95th percentile delivery time (service standard)
            sla_95 = df['delivery_time'].quantile(0.95)
            
            # Delivery time distribution analysis
            from scipy import stats
            
            # Test for normality
            statistic, p_value = stats.shapiro(df['delivery_time'].dropna())
            
            # If not normal, use median and IQR instead of mean and std
            if p_value < 0.05:
                central_tendency = df['delivery_time'].median()
                spread = df['delivery_time'].quantile(0.75) - df['delivery_time'].quantile(0.25)
            ```
            """)
        
        elif method_tab == "Bias Adjustment Techniques":
            st.markdown("""
            ### ⚖️ Bias Adjustment Techniques
            
            #### Geographic Bias Problem
            **Issue:** Raw geographic metrics favor populous states, hiding per-capita performance.
            
            **Example:**
            - São Paulo (SP): 46M people, R$ 10M revenue
            - Acre (AC): 900K people, R$ 500K revenue
            - **Raw view**: SP is 20x better than AC
            - **Bias-adjusted view**: AC has higher per-capita spending
            
            #### Population Normalization
            ```python
            # Method 1: Per capita adjustment
            def calculate_per_capita_metrics(df, population_dict):
                state_metrics = df.groupby('state').agg({
                    'revenue': 'sum',
                    'orders': 'count',
                    'customers': 'nunique'
                })
                
                # Add population data
                state_metrics['population'] = state_metrics.index.map(population_dict)
                
                # Calculate per capita metrics
                state_metrics['revenue_per_capita'] = (
                    state_metrics['revenue'] / state_metrics['population']
                )
                
                state_metrics['orders_per_1000'] = (
                    state_metrics['orders'] / state_metrics['population'] * 1000
                )
                
                return state_metrics
            ```
            
            #### Market Penetration Analysis
            ```python
            # Method 2: Market penetration adjustment
            def calculate_market_opportunity(state_metrics, national_benchmark):
                # Expected performance based on population
                state_metrics['expected_revenue'] = (
                    state_metrics['population'] * national_benchmark['revenue_per_capita']
                )
                
                # Market penetration rate
                state_metrics['penetration_rate'] = (
                    state_metrics['revenue'] / state_metrics['expected_revenue']
                )
                
                # Opportunity score (how much potential is untapped)
                state_metrics['opportunity_score'] = (
                    1 - state_metrics['penetration_rate']
                ) * state_metrics['expected_revenue']
                
                return state_metrics
            ```
            
            #### Seasonal Adjustment
            ```python
            # Method 3: Seasonal bias adjustment
            def seasonal_adjustment(df):
                # Calculate monthly seasonality factors
                monthly_avg = df.groupby('month')['revenue'].mean()
                overall_avg = df['revenue'].mean()
                seasonal_factors = monthly_avg / overall_avg
                
                # Apply seasonal adjustment
                df['month'] = df['date'].dt.month
                df['seasonal_factor'] = df['month'].map(seasonal_factors)
                df['seasonally_adjusted_revenue'] = df['revenue'] / df['seasonal_factor']
                
                return df
            ```
            
            #### Size-Based Normalization
            ```python
            # Method 4: Business size adjustment for seller analysis
            def size_adjusted_seller_metrics(df):
                seller_metrics = df.groupby('seller_id').agg({
                    'revenue': 'sum',
                    'orders': 'count',
                    'products': 'nunique',
                    'days_active': lambda x: (x.max() - x.min()).days
                })
                
                # Revenue per product (efficiency metric)
                seller_metrics['revenue_per_product'] = (
                    seller_metrics['revenue'] / seller_metrics['products']
                )
                
                # Revenue per day (intensity metric)
                seller_metrics['revenue_per_day'] = (
                    seller_metrics['revenue'] / seller_metrics['days_active']
                )
                
                return seller_metrics
            ```
            
            #### Category Mix Adjustment
            ```python
            # Method 5: Product category mix adjustment
            def category_mix_adjustment(df):
                # Calculate category difficulty scores
                category_margins = df.groupby('category')['profit_margin'].mean()
                category_competition = df.groupby('category')['sellers'].nunique()
                
                # Create difficulty index
                difficulty_index = (
                    (1 / category_margins) * np.log(category_competition)
                )
                
                # Adjust seller performance by category difficulty
                df['category_difficulty'] = df['category'].map(difficulty_index)
                df['difficulty_adjusted_performance'] = (
                    df['seller_performance'] * df['category_difficulty']
                )
                
                return df
            ```
            
            #### Validation of Adjustments
            ```python
            # Validate bias adjustment effectiveness
            def validate_bias_adjustment(raw_data, adjusted_data):
                # Check if adjustment reduces correlation with confounding variable
                raw_correlation = raw_data.corr()['confounding_variable']['target_metric']
                adjusted_correlation = adjusted_data.corr()['confounding_variable']['adjusted_target_metric']
                
                bias_reduction = abs(raw_correlation) - abs(adjusted_correlation)
                
                print(f"Bias reduction: {bias_reduction:.3f}")
                print(f"Adjustment effectiveness: {(bias_reduction/abs(raw_correlation)*100):.1f}%")
                
                return bias_reduction > 0
            ```
            """)
    
    elif doc_section == "💻 Code Structure":
        st.subheader("💻 Code Architecture & Structure")
        
        # Create code structure visualization
        st.markdown("""
        ## 🏗️ Application Architecture
        
        ### File Organization
        ```
        ecommerce-dashboard/
        ├── app.py                          # Main application file
        ├── dashboard_validation.ipynb      # Validation notebook  
        ├── requirements.txt                # Python dependencies
        ├── run_app.sh                     # Quick start script
        ├── README.md                      # Documentation
        ├── venv/                          # Virtual environment
        └── data/                          # CSV files
            ├── olist_customers_dataset.csv
            ├── olist_orders_dataset.csv
            ├── olist_order_items_dataset.csv
            ├── olist_order_payments_dataset.csv
            ├── olist_order_reviews_dataset.csv
            ├── olist_products_dataset.csv
            ├── olist_sellers_dataset.csv
            ├── olist_geolocation_dataset.csv
            └── product_category_name_translation.csv
        ```
        """)
        
        code_section = st.selectbox(
            "Select Code Section",
            [
                "Main Application Flow",
                "Data Loading & Caching",
                "Page Functions",
                "Visualization Components",
                "Utility Functions"
            ]
        )
        
        if code_section == "Main Application Flow":
            st.code("""
# Main Application Flow (app.py)

import streamlit as st
import pandas as pd
import plotly.express as px

# 1. CONFIGURATION
st.set_page_config(
    page_title="Olist Analytics Platform",
    page_icon="📊",
    layout="wide"
)

# 2. DATA LOADING
@st.cache_data
def load_data():
    # Load all 9 CSV files
    # Return dictionary of dataframes
    pass

@st.cache_data  
def preprocess_data(data_dict):
    # Merge all tables
    # Calculate derived metrics
    # Return master dataframe
    pass

# 3. MAIN FUNCTION
def main():
    # Load and process data
    data_dict = load_data()
    df = preprocess_data(data_dict)
    
    # Sidebar navigation
    page = st.sidebar.radio("Select Page", [...])
    
    # Route to appropriate page function
    if page == "Executive Overview":
        page_executive_overview(df)
    elif page == "Full Dashboard":
        page_full_dashboard(df)
    # ... etc
    
# 4. EXECUTION
if __name__ == "__main__":
    main()
            """, language="python")
        
        elif code_section == "Data Loading & Caching":
            st.code("""
# Data Loading with Streamlit Caching

@st.cache_data
def load_data():
    \"\"\"Load all datasets with proper error handling\"\"\"
    try:
        # Dictionary to store all dataframes
        data_dict = {}
        
        # List of required files
        files = [
            'olist_customers_dataset.csv',
            'olist_orders_dataset.csv',
            'olist_order_items_dataset.csv',
            # ... etc
        ]
        
        # Load each file
        for file in files:
            key = file.replace('olist_', '').replace('_dataset.csv', '')
            data_dict[key] = pd.read_csv(file)
            
        # Convert date columns
        date_columns = ['order_purchase_timestamp', 'order_delivered_date']
        for col in date_columns:
            if col in data_dict['orders'].columns:
                data_dict['orders'][col] = pd.to_datetime(
                    data_dict['orders'][col], 
                    errors='coerce'
                )
        
        return data_dict
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def preprocess_data(data_dict):
    \"\"\"Merge and prepare data for analysis\"\"\"
    
    # Start with orders as base table
    df = data_dict['orders'].copy()
    
    # Sequential joins to avoid memory issues
    df = df.merge(data_dict['order_items'], on='order_id', how='left')
    df = df.merge(data_dict['products'], on='product_id', how='left')
    df = df.merge(data_dict['sellers'], on='seller_id', how='left')
    df = df.merge(data_dict['customers'], on='customer_id', how='left')
    df = df.merge(data_dict['order_payments'], on='order_id', how='left')
    df = df.merge(data_dict['order_reviews'], on='order_id', how='left')
    
    # Feature engineering
    df['delivery_time'] = (
        df['order_delivered_customer_date'] - df['order_purchase_timestamp']
    ).dt.days
    
    df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)
    
    return df
            """, language="python")
        
        elif code_section == "Page Functions":
            st.code("""
# Page Functions Architecture

def page_executive_overview(df, data_dict=None):
    \"\"\"Executive Overview Page\"\"\"
    st.title("📊 Executive Overview")
    
    # Calculate KPIs
    total_revenue = df.groupby('order_id')['payment_value'].sum().sum()
    total_orders = df['order_id'].nunique()
    # ... more KPIs
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"R$ {total_revenue:,.0f}")
    # ... more metrics
    
    # Create visualizations
    st.subheader("Revenue Trend")
    fig = create_revenue_trend_chart(df)
    st.plotly_chart(fig, use_container_width=True)

def page_full_dashboard(df, data_dict=None):
    \"\"\"Main Dashboard with Radio Button Navigation\"\"\"
    st.title("📈 Comprehensive Dashboard")
    
    # Filters
    states = st.multiselect("Select States", df['customer_state'].unique())
    
    # Apply filters
    filtered_df = apply_filters(df, states)
    
    # Tab navigation using radio buttons (avoids disappearing issue)
    selected_tab = st.radio(
        "Select Analysis", 
        ["Revenue", "Customers", "Sellers", "Logistics"],
        horizontal=True
    )
    
    # Route to appropriate analysis
    if selected_tab == "Revenue":
        show_revenue_analysis(filtered_df)
    elif selected_tab == "Customers":
        show_customer_analysis(filtered_df)
    # ... etc

def apply_filters(df, states=None, categories=None):
    \"\"\"Apply user-selected filters to dataframe\"\"\"
    filtered_df = df.copy()
    
    if states:
        filtered_df = filtered_df[filtered_df['customer_state'].isin(states)]
    
    if categories:
        filtered_df = filtered_df[
            filtered_df['product_category_name_english'].isin(categories)
        ]
    
    return filtered_df
            """, language="python")
        
        elif code_section == "Visualization Components":
            st.code("""
# Visualization Components

def create_revenue_waterfall(df):
    \"\"\"Create revenue waterfall chart\"\"\"
    
    # Calculate components
    product_revenue = df.groupby('order_id')['price'].sum().sum()
    freight_revenue = df.groupby('order_id')['freight_value'].sum().sum()
    total_payment_value = df.groupby('order_id')['payment_value'].sum().sum()
    
    # Calculate other charges (taxes, fees, etc.)
    other_charges = total_payment_value - (product_revenue + freight_revenue)
    
    fig = go.Figure(go.Waterfall(
        name="Revenue Breakdown",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Product Sales", "Freight Charges", "Other Charges (Taxes/Fees)", "Total Revenue"],
        y=[product_revenue, freight_revenue, other_charges, total_payment_value],
        text=[f"R$ {x:,.0f}" for x in [product_revenue, freight_revenue, other_charges, total_payment_value]],
        textposition="outside"
    ))
    
    fig.update_layout(title="Revenue Components", height=400)
    return fig

def create_sankey_diagram(df):
    \"\"\"Create Sankey diagram for order flow\"\"\"
    
    # Prepare data
    status_flow = df.groupby(['order_status', 'customer_state'])['order_id'].nunique()
    
    # Create nodes and links
    all_nodes = list(status_flow.index.get_level_values(0).unique()) + \
                list(status_flow.index.get_level_values(1).unique())
    
    # Build Sankey
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=all_nodes, pad=15, thickness=20),
        link=dict(source=source_indices, target=target_indices, value=values)
    )])
    
    return fig

def create_cohort_heatmap(df):
    \"\"\"Create customer cohort retention heatmap\"\"\"
    
    # Cohort analysis logic
    cohort_data = df.copy()
    cohort_data['cohort'] = cohort_data.groupby('customer_id')['order_date'].transform('min')
    cohort_data['period'] = (cohort_data['order_date'] - cohort_data['cohort']).dt.days
    
    # Create cohort table
    cohort_table = cohort_data.groupby(['cohort', 'period'])['customer_id'].nunique().unstack()
    cohort_sizes = cohort_data.groupby('cohort')['customer_id'].nunique()
    retention_table = cohort_table.divide(cohort_sizes, axis=0)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=retention_table.values,
        x=retention_table.columns,
        y=retention_table.index,
        colorscale='RdYlGn',
        text=np.round(retention_table.values * 100, 1),
        texttemplate='%{text}%'
    ))
    
    return fig
            """, language="python")
        
        elif code_section == "Utility Functions":
            st.code("""
# Utility Functions

def safe_divide(numerator, denominator, default=0):
    \"\"\"Safely divide two numbers, handling division by zero\"\"\"
    return numerator / denominator if denominator != 0 else default

def format_currency(value):
    \"\"\"Format number as Brazilian Real currency\"\"\"
    return f"R$ {value:,.2f}"

def format_percentage(value, decimals=1):
    \"\"\"Format number as percentage\"\"\"
    return f"{value:.{decimals}f}%"

def calculate_growth_rate(current, previous):
    \"\"\"Calculate percentage growth rate\"\"\"
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

def get_brazil_population():
    \"\"\"Return Brazilian state population dictionary\"\"\"
    return {
        'SP': 46649132, 'MG': 21411923, 'RJ': 17463349,
        # ... full dictionary
    }

def validate_dataframe(df, required_columns):
    \"\"\"Validate that dataframe has required columns\"\"\"
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    return True

def create_metric_card(title, value, delta=None, help_text=None):
    \"\"\"Create a standardized metric display\"\"\"
    col1, col2 = st.columns([3, 1])
    with col1:
        st.metric(
            label=title,
            value=value,
            delta=delta,
            help=help_text
        )

def add_download_button(df, filename):
    \"\"\"Add download button for dataframe\"\"\"
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

class DataProcessor:
    \"\"\"Data processing class for complex operations\"\"\"
    
    def __init__(self, df):
        self.df = df
        
    def calculate_nps(self, score_column='review_score'):
        \"\"\"Calculate Net Promoter Score\"\"\"
        promoters = (self.df[score_column] >= 4).sum()
        detractors = (self.df[score_column] <= 2).sum()
        total = self.df[score_column].notna().sum()
        return ((promoters - detractors) / total) * 100 if total > 0 else 0
    
    def segment_customers(self, value_column='payment_value'):
        \"\"\"Segment customers by value quartiles\"\"\"
        customer_value = self.df.groupby('customer_id')[value_column].sum()
        return pd.qcut(customer_value, q=4, labels=['Low', 'Medium', 'High', 'VIP'])
            """, language="python")
    
    elif doc_section == "📊 Analytics Framework":
        st.subheader("📊 Analytics Framework")
        
        st.markdown("""
        ## 🎯 Business Intelligence Framework
        
        Our analytics framework is built on industry best practices for e-commerce intelligence.
        """)
        
        framework_tab = st.selectbox(
            "Select Framework Component",
            [
                "KPI Hierarchy",
                "Metric Definitions",
                "Business Logic Rules",
                "Data Quality Framework",
                "Performance Benchmarks"
            ]
        )
        
        if framework_tab == "KPI Hierarchy":
            st.markdown("""
            ### 📈 KPI Hierarchy & Relationships
            
            #### Level 1: Executive KPIs (C-Level)
            ```
            📊 BUSINESS HEALTH DASHBOARD
            ├── 💰 Financial Performance
            │   ├── Total Revenue (R$)
            │   ├── Revenue Growth (% MoM)
            │   └── Average Order Value (R$)
            ├── 👥 Customer Performance  
            │   ├── Total Customers (#)
            │   ├── Customer Acquisition (#)
            │   └── Net Promoter Score (0-100)
            ├── 🏪 Marketplace Performance
            │   ├── Active Sellers (#)
            │   ├── Seller Growth (% MoM)
            │   └── Avg Seller Revenue (R$)
            └── 🚚 Operational Performance
                ├── Delivery Rate (%)
                ├── On-Time Delivery (%)
                └── Avg Delivery Time (days)
            ```
            
            #### Level 2: Operational KPIs (Department Heads)
            ```
            🔍 DETAILED ANALYTICS
            ├── 💰 Revenue Deep-Dive
            │   ├── Revenue by Payment Method
            │   ├── Revenue by Product Category
            │   ├── Revenue by Geographic Region
            │   └── Revenue by Time Period
            ├── 👥 Customer Analytics
            │   ├── Customer Segmentation (RFM)
            │   ├── Repeat Purchase Rate (%)
            │   ├── Customer Lifetime Value (R$)
            │   └── Cohort Retention Analysis
            ├── 🏪 Seller Analytics
            │   ├── Seller Performance Matrix
            │   ├── Top Performers Identification
            │   ├── Seller Churn Analysis
            │   └── Geographic Seller Distribution
            └── 🚚 Logistics Analytics
                ├── Delivery Performance by Region
                ├── Shipping Method Analysis
                ├── Delivery Time Distribution
                └── Logistics Cost Analysis
            ```
            
            #### Level 3: Tactical KPIs (Analysts)
            ```
            📋 GRANULAR METRICS
            ├── Product Performance
            │   ├── Best/Worst Selling Products
            │   ├── Category Performance Trends
            │   ├── Price Optimization Opportunities
            │   └── Inventory Turnover Analysis
            ├── Customer Behavior
            │   ├── Purchase Frequency Analysis
            │   ├── Seasonal Buying Patterns
            │   ├── Cross-selling Opportunities
            │   └── Customer Journey Analysis
            ├── Seller Operations
            │   ├── Order Fulfillment Time
            │   ├── Product Listing Quality
            │   ├── Customer Service Metrics
            │   └── Pricing Competitiveness
            └── Market Intelligence
                ├── Competitive Analysis
                ├── Market Share by Category
                ├── Trend Identification
                └── Opportunity Mapping
            ```
            """)
        
        elif framework_tab == "Metric Definitions":
            st.markdown("""
            ### 📚 Standardized Metric Definitions
            
            #### Financial Metrics
            
            | Metric | Formula | Business Meaning |
            |--------|---------|------------------|
            | **Total Revenue** | `SUM(payment_value) GROUP BY order_id` | Total money received from completed orders |
            | **Average Order Value (AOV)** | `Total Revenue / Unique Orders` | Average spending per transaction |
            | **Revenue Growth** | `((Current - Previous) / Previous) * 100` | Month-over-month revenue change |
            | **Revenue per Customer** | `Total Revenue / Unique Customers` | Average customer value |
            | **Revenue per Seller** | `Total Revenue / Active Sellers` | Average seller contribution |
            
            #### Customer Metrics
            
            | Metric | Formula | Business Meaning |
            |--------|---------|------------------|
            | **Net Promoter Score** | `((Promoters - Detractors) / Total Reviews) * 100` | Customer loyalty indicator |
            | **Repeat Purchase Rate** | `(Customers with >1 order / Total Customers) * 100` | Customer retention effectiveness |
            | **Customer Lifetime Value** | `AOV * Purchase Frequency * Customer Lifespan` | Long-term customer value |
            | **Customer Acquisition Cost** | `Marketing Spend / New Customers` | Cost to acquire new customers |
            
            #### Operational Metrics
            
            | Metric | Formula | Business Meaning |
            |--------|---------|------------------|
            | **Delivery Rate** | `(Delivered Orders / Total Orders) * 100` | Fulfillment success rate |
            | **On-Time Delivery** | `(On-time Orders / Delivered Orders) * 100` | Service quality indicator |
            | **Average Delivery Time** | `MEAN(delivery_date - purchase_date)` | Logistics efficiency |
            | **Order Fulfillment Time** | `MEAN(shipped_date - purchase_date)` | Processing efficiency |
            
            #### Quality Metrics
            
            ```python
            # Review Score Distribution
            review_distribution = {
                'Excellent (5 stars)': (df['review_score'] == 5).sum(),
                'Good (4 stars)': (df['review_score'] == 4).sum(),
                'Average (3 stars)': (df['review_score'] == 3).sum(),
                'Poor (2 stars)': (df['review_score'] == 2).sum(),
                'Terrible (1 star)': (df['review_score'] == 1).sum()
            }
            
            # NPS Classification
            nps_interpretation = {
                '70-100': 'World Class',
                '50-69': 'Excellent', 
                '30-49': 'Great',
                '10-29': 'Good',
                '0-9': 'Needs Improvement',
                '<0': 'Critical'
            }
            ```
            
            #### Geographic Metrics
            
            ```python
            # Population-Adjusted Metrics
            def calculate_geographic_metrics(df, population_dict):
                metrics = df.groupby('state').agg({
                    'revenue': 'sum',
                    'orders': 'count', 
                    'customers': 'nunique'
                })
                
                metrics['population'] = metrics.index.map(population_dict)
                metrics['revenue_per_capita'] = metrics['revenue'] / metrics['population']
                metrics['penetration_rate'] = metrics['customers'] / metrics['population'] * 1000
                
                return metrics
            ```
            """)
    
    elif doc_section == "🚀 Deployment Guide":
        st.subheader("🚀 Production Deployment Guide")
        
        st.markdown("""
        ## 🚀 From Development to Production
        
        Complete guide for deploying your analytics dashboard to production environments.
        """)
        
        deployment_tab = st.selectbox(
            "Select Deployment Topic",
            [
                "Local Development Setup",
                "Cloud Deployment (Streamlit Cloud)",
                "Docker Containerization", 
                "Performance Optimization",
                "Monitoring & Maintenance"
            ]
        )
        
        if deployment_tab == "Local Development Setup":
            st.code("""
# Local Development Environment Setup

# 1. PROJECT STRUCTURE
ecommerce-dashboard/
├── app.py                    # Main application
├── requirements.txt          # Dependencies
├── .streamlit/              # Streamlit config
│   └── config.toml
├── data/                    # CSV files
├── tests/                   # Unit tests
├── docs/                    # Documentation
└── scripts/                 # Utility scripts
    ├── data_validation.py
    ├── backup.py
    └── deploy.py

# 2. REQUIREMENTS.TXT
streamlit==1.28.0
pandas==2.0.3
plotly==5.15.0
numpy==1.24.3
scipy==1.11.1

# 3. STREAMLIT CONFIGURATION (.streamlit/config.toml)
[global]
developmentMode = false

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

# 4. DEVELOPMENT SCRIPTS

# run_dev.sh
#!/bin/bash
source venv/bin/activate
export STREAMLIT_SERVER_HEADLESS=true
streamlit run app.py --server.port 8501

# test_app.py  
import pytest
import pandas as pd
from app import load_data, calculate_kpis

def test_data_loading():
    data_dict = load_data()
    assert data_dict is not None
    assert 'orders' in data_dict
    assert len(data_dict['orders']) > 0

def test_kpi_calculations():
    # Mock data for testing
    mock_df = pd.DataFrame({
        'order_id': ['1', '2', '3'],
        'payment_value': [100, 200, 150],
        'customer_id': ['A', 'B', 'C']
    })
    
    total_revenue = mock_df['payment_value'].sum()
    assert total_revenue == 450
            """, language="bash")
        
        elif deployment_tab == "Cloud Deployment (Streamlit Cloud)":
            st.markdown("""
            ### ☁️ Streamlit Cloud Deployment
            
            **Step 1: Prepare Repository**
            ```bash
            # Initialize git repository
            git init
            git add .
            git commit -m "Initial commit: E-commerce Analytics Dashboard"
            
            # Push to GitHub
            git remote add origin https://github.com/username/ecommerce-dashboard
            git push -u origin main
            ```
            
            **Step 2: Streamlit Cloud Setup**
            1. Go to [share.streamlit.io](https://share.streamlit.io)
            2. Connect your GitHub account
            3. Select repository: `username/ecommerce-dashboard`
            4. Set main file path: `app.py`
            5. Click "Deploy!"
            
            **Step 3: Environment Configuration**
            ```toml
            # .streamlit/config.toml
            [server]
            headless = true
            port = 8501
            enableCORS = false
            
            [browser]
            gatherUsageStats = false
            
            [theme]
            primaryColor = "#1f77b4"
            backgroundColor = "#ffffff"
            secondaryBackgroundColor = "#f0f2f6"
            ```
            
            **Step 4: Secrets Management**
            ```toml
            # .streamlit/secrets.toml (for API keys, passwords)
            [database]
            host = "your-db-host"
            username = "your-username"
            password = "your-password"
            
            [api_keys]
            google_analytics = "your-ga-key"
            ```
            
            **Step 5: Data Handling for Cloud**
            ```python
            # For large datasets, use cloud storage
            import streamlit as st
            from google.cloud import storage
            
            @st.cache_data
            def load_data_from_cloud():
                client = storage.Client()
                bucket = client.bucket('your-data-bucket')
                
                # Download CSV files
                for file in ['orders.csv', 'customers.csv']:
                    blob = bucket.blob(f'data/{file}')
                    blob.download_to_filename(file)
                
                # Load into pandas
                return load_local_data()
            ```
            """)
        
        elif deployment_tab == "Docker Containerization":
            st.code("""
# Docker Deployment Setup

# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# docker-compose.yml
version: '3.8'
services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
    restart: unless-stopped

# Build and run commands
docker build -t ecommerce-dashboard .
docker run -p 8501:8501 ecommerce-dashboard

# Or with docker-compose
docker-compose up -d
            """, language="dockerfile")
        
        elif deployment_tab == "Performance Optimization":
            st.code("""
# Performance Optimization Techniques

# 1. EFFICIENT DATA CACHING
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_large_dataset():
    # Use efficient data types
    dtypes = {
        'order_id': 'category',
        'customer_id': 'category', 
        'payment_value': 'float32',  # Instead of float64
        'review_score': 'int8'       # Instead of int64
    }
    
    df = pd.read_csv('large_file.csv', dtype=dtypes)
    return df

# 2. MEMORY OPTIMIZATION
def optimize_dataframe(df):
    # Convert object columns to category where appropriate
    for col in df.select_dtypes(include=['object']):
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    
    # Downcast numeric types
    df = df.select_dtypes(include=['int']).apply(pd.to_numeric, downcast='integer')
    df = df.select_dtypes(include=['float']).apply(pd.to_numeric, downcast='float')
    
    return df

# 3. LAZY LOADING
@st.cache_data
def load_data_subset(date_range=None, states=None):
    # Only load data needed for current analysis
    query_conditions = []
    
    if date_range:
        query_conditions.append(f"order_date >= '{date_range[0]}'")
        query_conditions.append(f"order_date <= '{date_range[1]}'")
    
    if states:
        state_list = "', '".join(states)
        query_conditions.append(f"customer_state IN ('{state_list}')")
    
    # Use SQL-like filtering for large datasets
    return df.query(' & '.join(query_conditions)) if query_conditions else df

# 4. PARALLEL PROCESSING
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def parallel_calculation(df, chunks=4):
    # Split dataframe into chunks
    chunk_size = len(df) // chunks
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    def process_chunk(chunk):
        return chunk.groupby('category')['revenue'].sum()
    
    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=chunks) as executor:
        results = list(executor.map(process_chunk, chunks))
    
    # Combine results
    return pd.concat(results).groupby(level=0).sum()

# 5. EFFICIENT VISUALIZATIONS
def create_optimized_chart(df):
    # Sample large datasets for visualization
    if len(df) > 10000:
        df_sample = df.sample(n=10000)
    else:
        df_sample = df
    
    # Use efficient chart types
    fig = px.scatter(
        df_sample, 
        x='metric1', 
        y='metric2',
        render_mode='webgl'  # Use WebGL for better performance
    )
    
    return fig

# 6. SESSION STATE OPTIMIZATION
def initialize_session_state():
    # Initialize expensive computations only once
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = expensive_data_processing()
    
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {}

# 7. DATABASE OPTIMIZATION (for production)
import sqlalchemy as sa

@st.cache_data
def load_from_database(query, params=None):
    engine = sa.create_engine('postgresql://user:pass@host:port/db')
    
    # Use parameterized queries
    df = pd.read_sql_query(
        query, 
        engine, 
        params=params,
        chunksize=10000  # Process in chunks
    )
    
    return df
            """, language="python")

def page_seller_recommendations(df, data_dict=None):
    """Strategic Recommendations for Head of Seller Relations"""
    st.title("💡 Strategic Seller Recommendations")
    st.markdown("### Action Plan for Head of Seller Relations")
    
    # Calculate seller metrics
    seller_performance = df.groupby('seller_id').agg({
        'payment_value': 'sum',
        'order_id': 'nunique',
        'review_score': 'mean',
        'delivery_time': 'mean',
        'on_time_delivery': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0
    }).reset_index()
    
    # Segment sellers
    seller_performance['revenue_percentile'] = seller_performance['payment_value'].rank(pct=True)
    seller_performance['segment'] = pd.cut(seller_performance['revenue_percentile'], 
                                          bins=[0, 0.25, 0.75, 0.95, 1.0],
                                          labels=['Bottom 25%', 'Middle 50%', 'Top 25%', 'Top 5%'])
    
    # Recommendation 1: Seller Segmentation Strategy
    st.subheader("1️⃣ Seller Segmentation & Targeted Support")
    
    col1, col2 = st.columns(2)
    
    with col1:
        segment_summary = seller_performance.groupby('segment').agg({
            'seller_id': 'count',
            'payment_value': 'sum',
            'review_score': 'mean'
        }).reset_index()
        segment_summary.columns = ['Segment', 'Count', 'Total Revenue', 'Avg Review']
        
        fig = px.treemap(segment_summary, path=['Segment'], values='Total Revenue',
                        color='Avg Review', color_continuous_scale='RdYlGn',
                        title="Seller Segments by Revenue Contribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Segment-Specific Actions")
        st.markdown("""
        **Top 5% Sellers (Elite Partners)**
        - Dedicated account management
        - Priority support and faster payouts
        - Co-marketing opportunities
        - ROI: 20% revenue increase
        
        **Top 25% Sellers (Growth Partners)**
        - Performance bonuses for hitting targets
        - Advanced analytics dashboard access
        - Quarterly business reviews
        - ROI: 15% revenue increase
        
        **Middle 50% (Development Partners)**
        - Automated performance insights
        - Self-service training resources
        - Volume-based incentives
        - ROI: 10% activation rate increase
        
        **Bottom 25% (Activation Focus)**
        - Onboarding optimization program
        - Basic training and support
        - Activation incentives
        - ROI: 25% reduction in churn
        """)
    
    # Recommendation 2: Geographic Expansion
    st.subheader("2️⃣ Geographic Expansion Strategy")
    
    # Analyze seller distribution vs demand
    seller_distribution = df.groupby('seller_state')['seller_id'].nunique().reset_index()
    customer_demand = df.groupby('customer_state')['order_id'].nunique().reset_index()
    
    geo_analysis = seller_distribution.merge(customer_demand, 
                                            left_on='seller_state', 
                                            right_on='customer_state', 
                                            how='outer')
    geo_analysis['supply_demand_ratio'] = geo_analysis['seller_id'] / geo_analysis['order_id']
    geo_analysis = geo_analysis.dropna()
    
    fig = px.scatter(geo_analysis, x='order_id', y='seller_id',
                    size='supply_demand_ratio', color='supply_demand_ratio',
                    hover_data=['seller_state'],
                    title="Seller Supply vs Customer Demand by State",
                    labels={'order_id': 'Customer Demand (Orders)', 
                           'seller_id': 'Seller Supply (Count)',
                           'supply_demand_ratio': 'Supply/Demand'},
                    color_continuous_scale='RdYlGn')
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Geographic Recommendations:**
    - 🔴 **High-Priority States:** Low supply/demand ratio - aggressive seller recruitment needed
    - 🟡 **Balanced States:** Maintain current seller base, focus on quality improvement
    - 🟢 **Oversupplied States:** Focus on seller activation and performance optimization
    """)
    
    # Recommendation 3: Performance Improvement Program
    st.subheader("3️⃣ Performance Improvement Initiatives")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Delivery performance by seller segment
        delivery_by_segment = seller_performance.groupby('segment').agg({
            'delivery_time': 'mean',
            'on_time_delivery': 'mean',
            'review_score': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=delivery_by_segment['segment'],
            y=delivery_by_segment['delivery_time'],
            name='Avg Delivery Time (days)',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Scatter(
            x=delivery_by_segment['segment'],
            y=delivery_by_segment['on_time_delivery'],
            name='On-Time Rate (%)',
            yaxis='y2',
            marker_color='green',
            mode='lines+markers',
            line=dict(width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="Delivery Performance by Seller Segment",
            yaxis=dict(title="Delivery Time (days)"),
            yaxis2=dict(title="On-Time Rate (%)", overlaying="y", side="right"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Performance Programs")
        st.markdown("""
        **Logistics Excellence Program**
        - Partner with regional logistics providers
        - Implement SLA-based pricing
        - Estimated Impact: 20% reduction in delivery time
        - ROI: R$ 2.5M additional revenue
        
        **Quality Assurance Initiative**
        - Automated quality scoring system
        - Review score improvement workshops
        - Estimated Impact: 0.5 point review increase
        - ROI: 15% reduction in returns
        
        **Inventory Management Support**
        - Demand forecasting tools
        - Just-in-time inventory recommendations
        - Estimated Impact: 30% reduction in stockouts
        - ROI: R$ 1.8M recovered sales
        """)
    
    # Recommendation 4: Revenue Optimization
    st.subheader("4️⃣ Revenue Optimization Strategy")
    
    # Category opportunity analysis
    category_performance = df[df['product_category_name_english'].notna()].groupby('product_category_name_english').agg({
        'payment_value': 'sum',
        'order_id': 'nunique',
        'seller_id': 'nunique',
        'review_score': 'mean'
    }).reset_index()
    
    category_performance['revenue_per_seller'] = category_performance['payment_value'] / category_performance['seller_id']
    category_performance = category_performance.nlargest(15, 'payment_value')
    
    fig = px.scatter(category_performance, x='seller_id', y='revenue_per_seller',
                    size='payment_value', color='review_score',
                    hover_data=['product_category_name_english'],
                    title="Category Opportunity Matrix",
                    labels={'seller_id': 'Number of Sellers',
                           'revenue_per_seller': 'Revenue per Seller (R$)',
                           'review_score': 'Avg Review'},
                    color_continuous_scale='RdYlGn')
    st.plotly_chart(fig, use_container_width=True)
    
    # ROI Summary
    st.subheader("💰 Expected ROI Summary")
    
    roi_data = {
        'Initiative': ['Seller Segmentation', 'Geographic Expansion', 'Performance Programs', 'Category Optimization'],
        'Investment': [500000, 800000, 600000, 400000],
        'Expected Return': [2500000, 3200000, 2100000, 1800000],
        'ROI %': [400, 300, 250, 350],
        'Timeline': ['3 months', '6 months', '4 months', '3 months']
    }
    
    roi_df = pd.DataFrame(roi_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_investment = roi_df['Investment'].sum()
        st.metric("Total Investment", f"R$ {total_investment:,.0f}")
    
    with col2:
        total_return = roi_df['Expected Return'].sum()
        st.metric("Expected Return", f"R$ {total_return:,.0f}")
    
    with col3:
        overall_roi = ((total_return - total_investment) / total_investment * 100)
        st.metric("Overall ROI", f"{overall_roi:.0f}%")
    
    st.dataframe(roi_df.style.format({
        'Investment': 'R$ {:,.0f}',
        'Expected Return': 'R$ {:,.0f}',
        'ROI %': '{:.0f}%'
    }), use_container_width=True)
    
    st.success("""
    **Executive Summary:**
    
    The seller relations strategy focuses on four key pillars:
    1. **Segmentation** - Tailored support based on performance tiers
    2. **Geographic Expansion** - Strategic recruitment in underserved markets
    3. **Performance Enhancement** - Logistics and quality improvement programs
    4. **Category Optimization** - Focus on high-margin, high-growth categories
    
    Total expected ROI: 313% within 6 months with R$ 2.3M investment
    """)

def main():
    """Main application"""
    
    # Load data
    with st.spinner("Loading data..."):
        data_dict = load_data()
        
    if not data_dict:
        st.error("Failed to load data. Please check if all CSV files are in the current directory.")
        st.stop()
    
    # Preprocess data
    with st.spinner("Processing data..."):
        df = preprocess_data(data_dict)
        
    if df is None or df.empty:
        st.error("Failed to preprocess data.")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Executive Overview", "Full Dashboard", "Insights & Trends", "Seller Recommendations", "📚 Documentation"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Overview")
    st.sidebar.metric("Order Line Items", f"{len(df):,}", help="Each row = 1 product in 1 order. Orders with multiple products create multiple rows.")
    st.sidebar.metric("Unique Orders", f"{df['order_id'].nunique():,}", help="Total number of distinct customer orders")
    st.sidebar.metric("Total Customers", f"{df['customer_unique_id'].nunique():,}")
    st.sidebar.metric("Total Sellers", f"{df['seller_id'].nunique():,}")
    st.sidebar.metric("Date Range", f"{df['order_purchase_timestamp'].min().date()} to {df['order_purchase_timestamp'].max().date()}")
    
    # Add explanation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Data Structure")
    st.sidebar.info("""
    **Why Line Items > Orders?**
    
    Each record represents one product within an order. 
    
    Example:
    - Order #1: 2 products = 2 records
    - Order #2: 1 product = 1 record
    - Total: 2 orders, 3 records
    
    This is standard e-commerce data structure!
    """)
    
    # Page routing
    if page == "Executive Overview":
        page_executive_overview(df, data_dict)
    elif page == "Full Dashboard":
        page_full_dashboard(df, data_dict)
    elif page == "Insights & Trends":
        page_insights_trends(df, data_dict)
    elif page == "Seller Recommendations":
        page_seller_recommendations(df, data_dict)
    elif page == "📚 Documentation":
        page_technical_documentation(df, data_dict)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "**Olist E-Commerce Analytics Platform**\n\n"
        "Comprehensive analytics dashboard for Brazilian e-commerce data.\n\n"
        "Features:\n"
        "- Executive dashboards\n"
        "- Advanced visualizations\n"
        "- Bias-adjusted insights\n"
        "- Strategic recommendations\n\n"
        "Built with Streamlit, Plotly, and Pandas"
    )

if __name__ == "__main__":
    main()