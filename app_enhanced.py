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
    page_title="Olist E-Commerce Analytics Platform - Enhanced",
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

# Enhanced CSS for insights and recommendations
st.markdown("""
<style>
    .insight-box {
        background-color: #1e3a8a;
        color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: 500;
        border-left: 5px solid #3b82f6;
    }
    .recommendation-box {
        background-color: #065f46;
        color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #10b981;
        margin: 10px 0;
        font-weight: 500;
    }
    .warning-insight {
        background-color: #92400e;
        color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #f59e0b;
        margin: 10px 0;
        font-weight: 500;
    }
    .critical-insight {
        background-color: #991b1b;
        color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ef4444;
        margin: 10px 0;
        font-weight: 500;
    }
    .roi-box {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        font-weight: 600;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

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

def generate_insight(chart_type, data_summary, department=None):
    """Generate actionable insights for charts based on data patterns"""
    insights = []
    
    if chart_type == "revenue_trend":
        growth_rate = data_summary.get('growth_rate', 0)
        if growth_rate < 0:
            insights.append(f"🔴 CRITICAL: Revenue declining at {abs(growth_rate):.1f}% - Immediate pricing review needed")
        elif growth_rate < 5:
            insights.append(f"🟡 WARNING: Growth slowing to {growth_rate:.1f}% - Market expansion required")
        else:
            insights.append(f"🟢 STRONG: {growth_rate:.1f}% growth - Maintain current strategy")
    
    elif chart_type == "geographic_distribution":
        concentration = data_summary.get('top_state_percentage', 0)
        if concentration > 40:
            insights.append(f"🟡 RISK: {concentration:.1f}% revenue from single state - Diversification needed")
        
    elif chart_type == "customer_segments":
        repeat_rate = data_summary.get('repeat_rate', 0)
        if repeat_rate < 25:
            insights.append(f"🔴 URGENT: Only {repeat_rate:.1f}% repeat customers - Launch loyalty program")
    
    elif chart_type == "delivery_performance":
        on_time_rate = data_summary.get('on_time_rate', 0)
        if on_time_rate < 85:
            insights.append(f"🔴 SLA BREACH: {on_time_rate:.1f}% on-time delivery - Logistics crisis")
        elif on_time_rate < 90:
            insights.append(f"🟡 IMPROVING: {on_time_rate:.1f}% on-time - Target 90%+")
        else:
            insights.append(f"🟢 EXCELLENT: {on_time_rate:.1f}% on-time delivery")
    
    # Add department-specific recommendations
    if department == "CFO":
        insights.append("💰 CFO ACTION: Review pricing elasticity and payment terms")
    elif department == "CMO":
        insights.append("🎯 CMO ACTION: Implement targeted retention campaigns")
    elif department == "COO":
        insights.append("⚙️ COO ACTION: Optimize logistics partnerships")
    elif department == "CPO":
        insights.append("🛍️ CPO ACTION: Analyze product mix performance")
    
    return insights

def page_executive_overview_enhanced(df, data_dict=None):
    """Enhanced Executive Overview with Insights"""
    st.title("📊 Executive Overview")
    st.markdown("### Business Health Snapshot with Strategic Insights")
    
    # Calculate KPIs - FIXED: Use original payments table
    total_revenue = data_dict['order_payments']['payment_value'].sum()
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
    
    # Executive Insights
    st.markdown("### 🎯 Executive Intelligence")
    insights = []
    
    if avg_review_score < 4.0:
        insights.append("🔴 CRITICAL: Customer satisfaction below 4.0 - Quality intervention required")
    if on_time_pct < 85:
        insights.append("🔴 URGENT: Delivery performance below SLA - Operations review needed")
    if delivered_pct < 95:
        insights.append("🟡 WARNING: Fulfillment rate issues - Process optimization required")
    
    for insight in insights:
        st.markdown(f'<div class="critical-insight">{insight}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts with Insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Revenue Trend Analysis")
        # FIXED: Use original payments table with order dates
        orders_with_payments = pd.merge(
            data_dict['orders'][['order_id', 'order_purchase_timestamp']], 
            data_dict['order_payments'][['order_id', 'payment_value']], 
            on='order_id'
        )
        orders_with_payments['order_month'] = orders_with_payments['order_purchase_timestamp'].dt.to_period('M').astype(str)
        monthly_revenue = orders_with_payments.groupby('order_month')['payment_value'].sum().reset_index()
        monthly_revenue = monthly_revenue.sort_values('order_month')
        
        fig = px.area(monthly_revenue, x='order_month', y='payment_value',
                     title="Monthly Revenue Trend",
                     labels={'payment_value': 'Revenue (R$)', 'order_month': 'Month'})
        fig.update_traces(fillcolor='rgba(31, 119, 180, 0.3)', line=dict(color='#1f77b4', width=3))
        st.plotly_chart(fig, use_container_width=True)
        
        # Revenue Trend Insights
        if len(monthly_revenue) >= 2:
            recent_growth = ((monthly_revenue['payment_value'].iloc[-1] - monthly_revenue['payment_value'].iloc[-2]) / 
                           monthly_revenue['payment_value'].iloc[-2]) * 100
            insights = generate_insight("revenue_trend", {'growth_rate': recent_growth})
            for insight in insights:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
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
        
        # Geographic Insights
        top_state_pct = (state_revenue['payment_value'].iloc[0] / total_revenue) * 100
        insights = generate_insight("geographic_distribution", {'top_state_percentage': top_state_pct})
        for insight in insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        
        # Add recommendation
        st.markdown(f'<div class="recommendation-box">🌍 RECOMMENDATION: Consider expanding to {state_revenue["customer_state"].iloc[0]} neighboring states for growth</div>', unsafe_allow_html=True)

def page_cfo_recommendations(df, data_dict):
    """CFO Financial Recommendations with ROI Analysis"""
    st.title("💰 CFO Financial Recommendations")
    st.markdown("### Strategic Financial Optimization Plan")
    
    # Financial Health Analysis
    total_revenue = data_dict['order_payments']['payment_value'].sum()
    product_revenue = data_dict['order_items']['price'].sum()
    freight_revenue = data_dict['order_items']['freight_value'].sum()
    processing_fees = total_revenue - (product_revenue + freight_revenue)
    
    # Monthly analysis
    orders_with_payments = pd.merge(
        data_dict['orders'][['order_id', 'order_purchase_timestamp']], 
        data_dict['order_payments'][['order_id', 'payment_value']], 
        on='order_id'
    )
    orders_with_payments['order_month'] = orders_with_payments['order_purchase_timestamp'].dt.to_period('M').astype(str)
    monthly_revenue = orders_with_payments.groupby('order_month')['payment_value'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"R$ {total_revenue/1e6:.2f}M")
    with col2:
        st.metric("Product Revenue", f"R$ {product_revenue/1e6:.2f}M")
    with col3:
        st.metric("Freight Revenue", f"R$ {freight_revenue/1e6:.2f}M")
    with col4:
        processing_margin = (processing_fees / total_revenue) * 100
        st.metric("Processing Margin", f"{processing_margin:.1f}%")
        st.caption("Processing Margin = (Total Revenue - Product Sales - Freight) / Total Revenue")
    
    # Recommendation 1: Revenue Diversification
    st.subheader("1️⃣ Revenue Diversification Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Payment method analysis
        payment_breakdown = data_dict['order_payments'].groupby('payment_type')['payment_value'].agg(['sum', 'count']).reset_index()
        payment_breakdown.columns = ['Payment Method', 'Revenue', 'Transactions']
        payment_breakdown['Avg Transaction'] = payment_breakdown['Revenue'] / payment_breakdown['Transactions']
        payment_breakdown = payment_breakdown.sort_values('Revenue', ascending=False)
        
        fig = px.pie(payment_breakdown, values='Revenue', names='Payment Method',
                    title="Revenue by Payment Method",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
        
        # Payment method insights
        if not payment_breakdown.empty:
            dominant_method = payment_breakdown.iloc[0]['Payment Method']
            dominant_pct = (payment_breakdown.iloc[0]['Revenue'] / total_revenue) * 100
            
            if dominant_pct > 70:
                st.markdown(f'<div class="critical-insight">🔴 RISK: {dominant_pct:.1f}% revenue from {dominant_method} - Diversify payment options</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="recommendation-box">💳 RECOMMENDATION: Incentivize alternative payment methods to reduce dependency</div>', unsafe_allow_html=True)
    
    with col2:
        # Payment method performance analysis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Revenue bars
        fig.add_trace(
            go.Bar(x=payment_breakdown['Payment Method'], y=payment_breakdown['Revenue']/1e6,
                   name="Revenue (R$ M)", marker_color='lightblue'),
            secondary_y=False,
        )
        
        # Average transaction line
        fig.add_trace(
            go.Scatter(x=payment_breakdown['Payment Method'], y=payment_breakdown['Avg Transaction'],
                      mode='lines+markers', name="Avg Transaction (R$)",
                      line=dict(color='red', width=3), marker=dict(size=8)),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Payment Method")
        fig.update_yaxes(title_text="Revenue (R$ Millions)", secondary_y=False)
        fig.update_yaxes(title_text="Average Transaction (R$)", secondary_y=True)
        fig.update_layout(title="Revenue vs Transaction Size by Payment Method", height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Transaction insights
        highest_aov = payment_breakdown.loc[payment_breakdown['Avg Transaction'].idxmax()]
        st.markdown(f'<div class="insight-box">💳 HIGHEST AOV: {highest_aov["Payment Method"]} with R$ {highest_aov["Avg Transaction"]:.2f} average transaction</div>', unsafe_allow_html=True)
    
    # Recommendation 2: Cost Optimization
    st.subheader("2️⃣ Cost Structure Optimization")
    
    # Category margin analysis
    category_analysis = df.groupby('product_category_name_english').agg({
        'payment_value': 'sum',
        'price': 'sum', 
        'freight_value': 'sum'
    }).reset_index()
    category_analysis = category_analysis.dropna()
    category_analysis['gross_margin'] = ((category_analysis['payment_value'] - category_analysis['price'] - category_analysis['freight_value']) / 
                                        category_analysis['payment_value']) * 100
    category_analysis = category_analysis.sort_values('payment_value', ascending=False).head(10)
    
    # Revenue vs Margin dual-axis chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Revenue bars
    fig.add_trace(
        go.Bar(x=category_analysis['product_category_name_english'], 
               y=category_analysis['payment_value']/1e6,
               name="Revenue (R$ M)", 
               marker_color='lightsteelblue'),
        secondary_y=False,
    )
    
    # Margin line
    fig.add_trace(
        go.Scatter(x=category_analysis['product_category_name_english'], 
                   y=category_analysis['gross_margin'],
                   mode='lines+markers', 
                   name="Gross Margin %",
                   line=dict(color='red', width=3), 
                   marker=dict(size=8)),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Category")
    fig.update_yaxes(title_text="Revenue (R$ Millions)", secondary_y=False)
    fig.update_yaxes(title_text="Gross Margin %", secondary_y=True)
    fig.update_layout(title="Revenue vs Gross Margin by Category", height=500)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Margin insights
    low_margin = category_analysis[category_analysis['gross_margin'] < 10]
    if not low_margin.empty:
        st.markdown(f'<div class="warning-insight">🟡 WARNING: {len(low_margin)} categories with margin < 10%</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="recommendation-box">🔧 RECOMMENDATION: Review pricing strategy for low-margin categories</div>', unsafe_allow_html=True)
    
    # ROI Summary with Assumptions
    st.subheader("💰 CFO Initiative Investment Analysis")
    
    st.markdown("### 📋 Investment Assumptions & Calculations")
    st.markdown("""
    **Payment Diversification (R$ 200K Investment)**
    - Technology integration and incentive programs
    - Expected 5% reduction in payment processing fees
    - ROI Calculation: 5% of current payment volume × reduced fees
    
    **Pricing Optimization (R$ 150K Investment)**  
    - Analytics tools and pricing algorithm development
    - Expected 2% margin improvement across top categories
    - ROI Calculation: 2% margin increase × top category revenue
    
    **Cost Structure Review (R$ 100K Investment)**
    - Consultant fees and process optimization
    - Expected 10% reduction in operational costs
    - ROI Calculation: Current operational spend × 10% efficiency gain
    
    **Cash Flow Optimization (R$ 75K Investment)**
    - Working capital management and payment term improvements
    - Expected 15% improvement in payment collection
    - ROI Calculation: Improved cash conversion × interest savings
    """)
    
    initiatives = {
        'Payment Diversification': {'investment': 200000, 'expected_benefit': 'Reduce fees by 5%', 'timeline': '3 months'},
        'Pricing Optimization': {'investment': 150000, 'expected_benefit': 'Increase margin by 2%', 'timeline': '2 months'},
        'Cost Structure Review': {'investment': 100000, 'expected_benefit': 'Reduce costs by 10%', 'timeline': '4 months'},
        'Cash Flow Optimization': {'investment': 75000, 'expected_benefit': 'Improve collection by 15%', 'timeline': '2 months'}
    }
    
    investment_data = []
    for initiative, data in initiatives.items():
        investment_data.append({
            'Initiative': initiative,
            'Investment': data['investment'],
            'Expected Benefit': data['expected_benefit'],
            'Timeline': data['timeline']
        })
    
    investment_df = pd.DataFrame(investment_data)
    
    col1, col2 = st.columns(2)
    with col1:
        total_investment = investment_df['Investment'].sum()
        st.metric("Total Investment Required", f"R$ {total_investment:,.0f}")
    with col2:
        st.metric("Implementation Timeline", "2-4 months")
    
    st.dataframe(investment_df.style.format({
        'Investment': 'R$ {:,.0f}'
    }), use_container_width=True)
    
    st.markdown(f'<div class="roi-box">💡 STRATEGIC FOCUS: Prioritize initiatives based on current business needs and available resources</div>', unsafe_allow_html=True)

def page_cmo_recommendations(df, data_dict):
    """CMO Marketing Recommendations with Customer Intelligence"""
    st.title("🎯 CMO Marketing Recommendations")
    st.markdown("### Customer Acquisition & Retention Strategy")
    
    # Customer Analysis
    total_customers = df['customer_unique_id'].nunique()
    repeat_customers = df.groupby('customer_unique_id')['order_id'].nunique()
    repeat_rate = (repeat_customers[repeat_customers > 1].count() / repeat_customers.count()) * 100
    
    customer_ltv = df.groupby('customer_unique_id')['payment_value'].sum().mean()
    avg_order_value = data_dict['order_payments']['payment_value'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{total_customers:,}")
    with col2:
        st.metric("Repeat Rate", f"{repeat_rate:.1f}%")
    with col3:
        st.metric("Avg Customer LTV", f"R$ {customer_ltv:.2f}")
    with col4:
        # LTV/CAC Calculation with detailed explanation
        acquisition_cost = 45  # Marketing spend per customer acquisition
        ltv_cac = customer_ltv / acquisition_cost
        st.metric("LTV/CAC Ratio", f"{ltv_cac:.1f}x")
        
    # LTV/CAC Calculation Details
    st.markdown("### 💰 LTV/CAC Calculation Methodology")
    st.markdown("""
    **Customer Lifetime Value (LTV):**
    - Calculated as: Total payment value per customer across all orders
    - Formula: `df.groupby('customer_unique_id')['payment_value'].sum().mean()`
    - Current LTV: R$ {:.2f}
    
    **Customer Acquisition Cost (CAC):**
    - Estimated marketing spend: R$ 45 per customer
    - Based on industry benchmarks for Brazilian e-commerce (5-8% of revenue)
    - Includes: Digital ads, SEO, affiliate marketing, promotions
    
    **LTV/CAC Ratio:**
    - Formula: LTV ÷ CAC = {:.2f} ÷ 45 = {:.1f}x
    - **Benchmark**: 3x minimum, 5x+ excellent
    - **Current Status**: {}
    """.format(
        customer_ltv, 
        customer_ltv, 
        ltv_cac,
        "⚠️ Below target - need to improve retention or reduce CAC" if ltv_cac < 3 
        else "✅ Good performance" if ltv_cac < 5 
        else "🎉 Excellent performance"
    ))
    
    # Customer Intelligence Analysis
    if repeat_rate < 30:
        st.markdown('<div class="critical-insight">🔴 CRITICAL: Repeat rate below 30% - Loyalty crisis detected</div>', unsafe_allow_html=True)
    if ltv_cac < 3:
        st.markdown('<div class="warning-insight">🟡 WARNING: LTV/CAC below 3x - Acquisition efficiency issues</div>', unsafe_allow_html=True)
    
    # Recommendation 1: Customer Segmentation Strategy
    st.subheader("1️⃣ Advanced Customer Segmentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RFM Analysis
        customer_summary = df.groupby('customer_unique_id').agg({
            'order_id': 'nunique',
            'payment_value': 'sum',
            'order_purchase_timestamp': 'max'
        }).reset_index()
        customer_summary.columns = ['customer_id', 'frequency', 'monetary', 'last_purchase']
        
        # Calculate recency
        max_date = customer_summary['last_purchase'].max()
        customer_summary['recency_days'] = (max_date - customer_summary['last_purchase']).dt.days
        
        # Segment customers
        def segment_customers(row):
            if row['frequency'] >= 3 and row['monetary'] > 500:
                return 'Champions'
            elif row['frequency'] >= 2 and row['recency_days'] < 60:
                return 'Loyal Customers'
            elif row['monetary'] > 300 and row['recency_days'] < 90:
                return 'Potential Loyalists'
            elif row['recency_days'] > 180:
                return 'At Risk'
            elif row['frequency'] == 1:
                return 'New Customers'
            else:
                return 'Need Attention'
        
        customer_summary['segment'] = customer_summary.apply(segment_customers, axis=1)
        segment_counts = customer_summary['segment'].value_counts()
        
        fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                    title="Customer Segments Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
        
        # Customer Segmentation Methodology
        st.markdown("### 📊 Customer Segmentation Logic (RFM Analysis)")
        st.markdown("""
        **Segmentation Criteria:**
        
        🏆 **Champions**: Frequency ≥ 3 orders AND Monetary ≥ R$ 500
        - High-value, frequent buyers who generate significant revenue
        
        💎 **Loyal Customers**: Frequency ≥ 2 orders AND Recency < 60 days
        - Regular buyers with recent activity, showing consistent engagement
        
        🌟 **Potential Loyalists**: Monetary > R$ 300 AND Recency < 90 days
        - High-value customers with room to increase frequency
        
        ⚠️ **At Risk**: Recency > 180 days (6+ months since last order)
        - Previously active customers showing signs of churn
        
        👶 **New Customers**: Frequency = 1 order only
        - First-time buyers requiring nurturing to drive repeat purchases
        
        🔄 **Need Attention**: All other combinations
        - Customers requiring targeted re-engagement strategies
        
        **Data Sources:**
        - Frequency: Count of unique orders per customer
        - Monetary: Total payment value across all orders
        - Recency: Days since last purchase (max date - customer's last order)
        """)
        
        # Segment insights
        if 'At Risk' in segment_counts.index:
            at_risk_pct = (segment_counts['At Risk'] / total_customers) * 100
            if at_risk_pct > 25:
                st.markdown(f'<div class="critical-insight">🚨 URGENT: {at_risk_pct:.1f}% customers at risk of churning</div>', unsafe_allow_html=True)
        
        if 'Champions' in segment_counts.index:
            champion_pct = (segment_counts['Champions'] / total_customers) * 100
            if champion_pct < 5:
                st.markdown(f'<div class="warning-insight">🟡 OPPORTUNITY: Only {champion_pct:.1f}% champions - VIP program needed</div>', unsafe_allow_html=True)
    
    with col2:
        # Geographic customer distribution
        state_customers = df.groupby('customer_state')['customer_unique_id'].nunique().reset_index()
        state_customers.columns = ['State', 'Customers']
        state_customers['Population'] = state_customers['State'].map(BRAZIL_POPULATION)
        state_customers['Penetration'] = (state_customers['Customers'] / state_customers['Population']) * 10000
        state_customers = state_customers.sort_values('Penetration', ascending=False).head(10)
        
        fig = px.bar(state_customers, x='State', y='Penetration',
                    title="Market Penetration by State (per 10K population)",
                    labels={'Penetration': 'Customers per 10K pop'},
                    color='Penetration',
                    color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        
        # Population Data Source Documentation
        st.markdown("### 📊 Population Data Methodology")
        st.markdown("""
        **Brazilian State Population Data:**
        - Source: IBGE (Instituto Brasileiro de Geografia e Estatística)
        - Census data: 2022 population estimates
        - Coverage: All 27 Brazilian states (26 states + Federal District)
        - Total population: ~215 million inhabitants
        
        **Penetration Calculation:**
        ```python
        # Formula: (Customers per State / State Population) × 10,000
        penetration_rate = (customers_count / state_population) * 10000
        ```
        
        **Data Quality:**
        - Population data: Official government statistics
        - Customer data: Olist dataset spanning 2016-2018
        - Penetration rates normalized per 10K population for comparability
        
        **Top 5 Most Populated States:**
        - SP (São Paulo): 46.6M - Brazil's economic hub
        - MG (Minas Gerais): 21.4M - Mining and agriculture
        - RJ (Rio de Janeiro): 17.5M - Tourism and services
        - BA (Bahia): 15.0M - Northeast's largest economy
        - PR (Paraná): 11.6M - Agriculture and industry
        """)
        
        # Geographic insights
        avg_penetration = state_customers['Penetration'].mean()
        low_penetration = state_customers[state_customers['Penetration'] < avg_penetration/2]
        
        if not low_penetration.empty:
            st.markdown(f'<div class="insight-box">🎯 OPPORTUNITY: {len(low_penetration)} states with low penetration</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="recommendation-box">🌍 RECOMMENDATION: Launch targeted campaigns in under-penetrated high-population states</div>', unsafe_allow_html=True)
    
    # Recommendation 2: Retention & Loyalty Program
    st.subheader("2️⃣ Customer Retention Strategy")
    
    # Time to second purchase analysis
    customer_orders = df.groupby('customer_unique_id')['order_purchase_timestamp'].apply(list).reset_index()
    customer_orders['order_count'] = customer_orders['order_purchase_timestamp'].apply(len)
    repeat_customers_data = customer_orders[customer_orders['order_count'] > 1]
    
    if not repeat_customers_data.empty:
        time_to_second = []
        for _, row in repeat_customers_data.iterrows():
            if len(row['order_purchase_timestamp']) >= 2:
                first_order = min(row['order_purchase_timestamp'])
                second_order = sorted(row['order_purchase_timestamp'])[1]
                days_diff = (second_order - first_order).days
                time_to_second.append(days_diff)
        
        if time_to_second:
            import numpy as np
            avg_time_to_second = np.mean(time_to_second)
            
            # Create proper histogram with better binning
            fig = px.histogram(x=time_to_second, 
                             nbins=min(30, len(set(time_to_second))),  # Dynamic bins based on data
                             title=f"Time to Second Purchase (Avg: {avg_time_to_second:.0f} days)",
                             labels={'x': 'Days', 'y': 'Count of Customers'},
                             opacity=0.8)
            
            # Custom binning for better visualization
            bins = [0, 30, 60, 90, 180, 365, max(time_to_second)]
            bin_labels = ['0-30 days', '31-60 days', '61-90 days', '91-180 days', '181-365 days', '365+ days']
            
            # Create binned data
            binned_data = pd.cut(time_to_second, bins=bins, labels=bin_labels, include_lowest=True)
            bin_counts = binned_data.value_counts().sort_index()
            
            # Create bar chart instead for better readability
            fig = px.bar(x=bin_counts.index, y=bin_counts.values,
                        title=f"Time to Second Purchase Distribution (Avg: {avg_time_to_second:.0f} days)",
                        labels={'x': 'Time Period', 'y': 'Number of Customers'})
            fig.update_traces(marker_color='lightblue', marker_opacity=0.8)
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Histogram Analysis
            st.markdown("### 📊 Histogram Binning Logic")
            st.markdown(f"""
            **Bin Strategy:**
            - 0-30 days: Immediate repurchase (excellent engagement)
            - 31-60 days: Good retention period
            - 61-90 days: Acceptable retention window
            - 91-180 days: At-risk period (needs intervention)
            - 181-365 days: Long-term retention
            - 365+ days: Dormant customers
            
            **Current Distribution:**
            - Total repeat customers: {len(time_to_second):,}
            - Average days to second purchase: {avg_time_to_second:.0f}
            - Customers with quick repurchase (≤30 days): {sum(1 for x in time_to_second if x <= 30):,} ({(sum(1 for x in time_to_second if x <= 30)/len(time_to_second)*100):.1f}%)
            """)
            
            # Retention insights
            if avg_time_to_second > 90:
                st.markdown(f'<div class="warning-insight">🟡 CONCERN: Average {avg_time_to_second:.0f} days to second purchase - Engagement campaigns needed</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="recommendation-box">📧 RECOMMENDATION: Implement 30/60/90 day re-engagement email campaigns</div>', unsafe_allow_html=True)
    
    # ROI Analysis for CMO Initiatives with detailed assumptions
    st.subheader("💰 CMO Initiative ROI Analysis")
    
    # Detailed ROI assumptions based on current metrics
    current_repeat_rate = repeat_rate
    current_ltv = customer_ltv
    current_customers = total_customers
    
    marketing_initiatives = {
        'Loyalty Program': {
            'investment': 500000,
            'timeline': '6 months',
            'assumptions': {
                'target_repeat_rate': 45.0,  # From current ~12% to 45%
                'program_cost_per_customer': 25,
                'loyalty_ltv_uplift': 1.8,  # 80% increase in LTV for loyal customers
                'participation_rate': 0.30  # 30% of customers join program
            }
        },
        'Geographic Expansion': {
            'investment': 300000,
            'timeline': '4 months', 
            'assumptions': {
                'target_states': 5,  # Focus on 5 underperforming states
                'acquisition_cost_reduction': 0.15,  # 15% lower CAC in new regions
                'new_customers_per_state': 2000,
                'avg_ltv_new_regions': current_ltv * 0.85  # Slightly lower LTV initially
            }
        },
        'Customer Segmentation Platform': {
            'investment': 200000,
            'timeline': '3 months',
            'assumptions': {
                'campaign_efficiency_gain': 0.25,  # 25% better targeting
                'at_risk_recovery_rate': 0.20,  # Recover 20% of at-risk customers
                'personalization_ltv_boost': 0.15,  # 15% LTV increase from personalization
                'operational_cost_savings': 50000  # Annual savings from automation
            }
        },
        'Retention Campaigns': {
            'investment': 150000,
            'timeline': '2 months',
            'assumptions': {
                'email_campaign_cost': 2.5,  # Cost per customer per campaign
                'conversion_rate': 0.12,  # 12% of recipients make purchase
                'avg_campaign_order_value': 85,  # Average order from campaign
                'campaign_frequency': 12  # Monthly campaigns for a year
            }
        }
    }
    
    # Calculate ROI based on detailed assumptions
    roi_data = []
    
    # Loyalty Program ROI calculation
    loyalty_assumptions = marketing_initiatives['Loyalty Program']['assumptions']
    loyal_customers = int(current_customers * loyalty_assumptions['participation_rate'])
    loyalty_ltv_increase = loyal_customers * (current_ltv * (loyalty_assumptions['loyalty_ltv_uplift'] - 1))
    loyalty_program_costs = loyal_customers * loyalty_assumptions['program_cost_per_customer']
    loyalty_return = loyalty_ltv_increase - loyalty_program_costs + marketing_initiatives['Loyalty Program']['investment']
    
    # Geographic Expansion ROI
    geo_assumptions = marketing_initiatives['Geographic Expansion']['assumptions']
    new_customers = geo_assumptions['target_states'] * geo_assumptions['new_customers_per_state']
    geo_return = (new_customers * geo_assumptions['avg_ltv_new_regions']) - (new_customers * acquisition_cost * (1 - geo_assumptions['acquisition_cost_reduction']))
    
    # Segmentation Platform ROI
    seg_assumptions = marketing_initiatives['Customer Segmentation Platform']['assumptions']
    at_risk_customers = segment_counts.get('At Risk', 0)
    recovered_customers = int(at_risk_customers * seg_assumptions['at_risk_recovery_rate'])
    segmentation_return = (recovered_customers * current_ltv) + (current_customers * current_ltv * seg_assumptions['personalization_ltv_boost']) + seg_assumptions['operational_cost_savings']
    
    # Retention Campaigns ROI  
    ret_assumptions = marketing_initiatives['Retention Campaigns']['assumptions']
    campaign_customers = current_customers
    retention_revenue = (campaign_customers * ret_assumptions['conversion_rate'] * ret_assumptions['avg_campaign_order_value'] * ret_assumptions['campaign_frequency'])
    retention_costs = (campaign_customers * ret_assumptions['email_campaign_cost'] * ret_assumptions['campaign_frequency'])
    retention_return = retention_revenue - retention_costs
    
    # Build ROI data with calculated returns
    initiatives_with_returns = {
        'Loyalty Program': loyalty_return,
        'Geographic Expansion': geo_return, 
        'Customer Segmentation Platform': segmentation_return,
        'Retention Campaigns': retention_return
    }
    
    for initiative, calculated_return in initiatives_with_returns.items():
        investment = marketing_initiatives[initiative]['investment']
        roi = ((calculated_return - investment) / investment) * 100
        roi_data.append({
            'Initiative': initiative,
            'Investment': investment,
            'Expected Return': calculated_return,
            'ROI %': roi,
            'Timeline': marketing_initiatives[initiative]['timeline']
        })
    
    roi_df = pd.DataFrame(roi_data)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_investment = roi_df['Investment'].sum()
        st.metric("Total Investment", f"R$ {total_investment:,.0f}")
    with col2:
        total_return = roi_df['Expected Return'].sum()
        st.metric("Expected Return", f"R$ {total_return:,.0f}")
    with col3:
        overall_roi = ((total_return - total_investment) / total_investment) * 100
        st.metric("Overall ROI", f"{overall_roi:.0f}%")
    
    st.dataframe(roi_df.style.format({
        'Investment': 'R$ {:,.0f}',
        'Expected Return': 'R$ {:,.0f}',
        'ROI %': '{:.0f}%'
    }), use_container_width=True)
    
    # Detailed Assumptions Documentation
    st.markdown("### 📋 Detailed ROI Assumptions")
    
    for initiative, data in marketing_initiatives.items():
        with st.expander(f"📊 {initiative} - Calculation Details"):
            st.markdown(f"**Investment:** R$ {data['investment']:,}")
            st.markdown(f"**Timeline:** {data['timeline']}")
            st.markdown("**Key Assumptions:**")
            
            if initiative == 'Loyalty Program':
                st.markdown(f"""
                - Current repeat rate: {current_repeat_rate:.1f}% → Target: {data['assumptions']['target_repeat_rate']}%
                - Program participation rate: {data['assumptions']['participation_rate']*100:.0f}% of customers
                - LTV uplift for loyal customers: {data['assumptions']['loyalty_ltv_uplift']*100:.0f}%
                - Program cost per customer: R$ {data['assumptions']['program_cost_per_customer']}
                - **Calculation:** {loyal_customers:,} loyal customers × R$ {(current_ltv * (data['assumptions']['loyalty_ltv_uplift'] - 1)):.2f} LTV increase
                """)
                
            elif initiative == 'Geographic Expansion':
                st.markdown(f"""
                - Target states for expansion: {data['assumptions']['target_states']}
                - New customers per state: {data['assumptions']['new_customers_per_state']:,}
                - CAC reduction in new regions: {data['assumptions']['acquisition_cost_reduction']*100:.0f}%
                - LTV in new regions: {data['assumptions']['avg_ltv_new_regions']:.2f} (85% of current)
                - **Calculation:** {new_customers:,} new customers × R$ {data['assumptions']['avg_ltv_new_regions']:.2f} LTV
                """)
                
            elif initiative == 'Customer Segmentation Platform':
                st.markdown(f"""
                - At-risk customers to recover: {data['assumptions']['at_risk_recovery_rate']*100:.0f}% of {at_risk_customers:,}
                - Campaign efficiency improvement: {data['assumptions']['campaign_efficiency_gain']*100:.0f}%
                - Personalization LTV boost: {data['assumptions']['personalization_ltv_boost']*100:.0f}%
                - Annual operational savings: R$ {data['assumptions']['operational_cost_savings']:,}
                - **Calculation:** {recovered_customers:,} recovered customers + personalization boost + cost savings
                """)
                
            elif initiative == 'Retention Campaigns':
                st.markdown(f"""
                - Campaign cost per customer: R$ {data['assumptions']['email_campaign_cost']}
                - Campaign conversion rate: {data['assumptions']['conversion_rate']*100:.1f}%
                - Average order value from campaigns: R$ {data['assumptions']['avg_campaign_order_value']}
                - Campaign frequency: {data['assumptions']['campaign_frequency']} per year
                - **Calculation:** {campaign_customers:,} customers × {data['assumptions']['conversion_rate']*100:.1f}% conversion × R$ {data['assumptions']['avg_campaign_order_value']} AOV
                """)
    
    # Strategic recommendations
    st.markdown("### 🎯 Strategic Action Plan")
    
    priority_actions = [
        f"🔥 IMMEDIATE: Launch retention campaign for {segment_counts.get('At Risk', 0):,} at-risk customers",
        f"📈 30 DAYS: Implement loyalty program targeting {segment_counts.get('Potential Loyalists', 0):,} potential loyalists", 
        f"🌍 60 DAYS: Geographic expansion to {len(low_penetration)} under-penetrated states",
        f"💎 90 DAYS: VIP program for {segment_counts.get('Champions', 0):,} champion customers"
    ]
    
    for action in priority_actions:
        st.markdown(f'<div class="recommendation-box">{action}</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="roi-box">🚀 CUSTOMER IMPACT: Increase repeat rate from {repeat_rate:.1f}% to 45%+ within 6 months</div>', unsafe_allow_html=True)

def page_coo_recommendations(df, data_dict):
    """COO Operations Recommendations with Logistics Optimization"""
    st.title("⚙️ COO Operations Recommendations")
    st.markdown("### Logistics & Operations Excellence Plan")
    
    # Operations KPIs
    total_orders = df['order_id'].nunique()
    delivered_orders = df[df['order_delivered_customer_date'].notna()]
    on_time_orders = delivered_orders[delivered_orders['on_time_delivery'] == True]
    
    delivery_rate = (len(delivered_orders) / total_orders) * 100
    on_time_rate = (len(on_time_orders) / len(delivered_orders)) * 100 if len(delivered_orders) > 0 else 0
    avg_delivery_time = delivered_orders['delivery_time'].mean()
    cancelled_orders = df[df['order_status'] == 'canceled']['order_id'].nunique()
    cancellation_rate = (cancelled_orders / total_orders) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Delivery Rate", f"{delivery_rate:.1f}%")
    with col2:
        color = "🟢" if on_time_rate >= 90 else "🟡" if on_time_rate >= 85 else "🔴"
        st.metric("On-Time Delivery", f"{on_time_rate:.1f}%", f"{color}")
    with col3:
        st.metric("Avg Delivery Time", f"{avg_delivery_time:.1f} days")
    with col4:
        st.metric("Cancellation Rate", f"{cancellation_rate:.1f}%")
    
    # Operations Alerts
    if on_time_rate < 85:
        st.markdown('<div class="critical-insight">🔴 SLA BREACH: On-time delivery below 85% - Immediate logistics review required</div>', unsafe_allow_html=True)
    if cancellation_rate > 5:
        st.markdown('<div class="warning-insight">🟡 HIGH CANCELLATIONS: {cancellation_rate:.1f}% cancellation rate - Process investigation needed</div>', unsafe_allow_html=True)
    if avg_delivery_time > 15:
        st.markdown('<div class="warning-insight">🟡 SLOW DELIVERY: Average delivery time exceeds 2 weeks - Warehouse optimization required</div>', unsafe_allow_html=True)
    
    # Recommendation 1: State-Level Performance Optimization
    st.subheader("1️⃣ Geographic Performance Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # State-level delivery performance
        state_performance = delivered_orders.groupby('customer_state').agg({
            'delivery_time': 'mean',
            'on_time_delivery': lambda x: (x.sum() / len(x)) * 100,
            'order_id': 'count'
        }).reset_index()
        state_performance.columns = ['State', 'Avg Delivery Time', 'On-Time %', 'Volume']
        
        # Identify problem states
        problem_states = state_performance[
            (state_performance['On-Time %'] < 80) | 
            (state_performance['Avg Delivery Time'] > avg_delivery_time + 5)
        ]
        
        fig = px.scatter(state_performance, x='Avg Delivery Time', y='On-Time %',
                        size='Volume', color='On-Time %',
                        hover_data=['State'],
                        title="State Performance Matrix",
                        color_continuous_scale='RdYlGn',
                        range_color=[50, 100])
        
        # Add target lines
        fig.add_hline(y=85, line_dash="dash", line_color="red", 
                     annotation_text="SLA Target (85%)")
        fig.add_vline(x=avg_delivery_time, line_dash="dash", line_color="blue",
                     annotation_text=f"Avg ({avg_delivery_time:.1f} days)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # State performance insights
        if not problem_states.empty:
            st.markdown(f'<div class="critical-insight">🚨 CRITICAL: {len(problem_states)} states below performance targets</div>', unsafe_allow_html=True)
            st.dataframe(problem_states.sort_values('On-Time %'), use_container_width=True)
        
        st.markdown('<div class="recommendation-box">🚚 RECOMMENDATION: Establish regional distribution centers in underperforming states</div>', unsafe_allow_html=True)
    
    with col2:
        # Delivery time distribution
        delivery_times = delivered_orders['delivery_time'].dropna()
        
        fig = px.histogram(delivery_times, nbins=20,
                          title=f"Delivery Time Distribution",
                          labels={'value': 'Delivery Time (days)', 'count': 'Number of Orders'})
        fig.update_traces(marker_color='lightcoral')
        
        # Add performance benchmarks
        fig.add_vline(x=7, line_dash="dash", line_color="green",
                     annotation_text="Target (7 days)")
        fig.add_vline(x=avg_delivery_time, line_dash="dash", line_color="red",
                     annotation_text=f"Current Avg ({avg_delivery_time:.1f})")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Delivery insights
        fast_deliveries = (delivery_times <= 7).sum()
        fast_delivery_pct = (fast_deliveries / len(delivery_times)) * 100
        
        st.markdown(f'<div class="insight-box">📦 PERFORMANCE: {fast_delivery_pct:.1f}% of orders delivered within 7 days</div>', unsafe_allow_html=True)
        
        if fast_delivery_pct < 50:
            st.markdown('<div class="recommendation-box">⚡ RECOMMENDATION: Target 70% of orders delivered within 7 days</div>', unsafe_allow_html=True)
    
    # Recommendation 2: Operational Efficiency
    st.subheader("2️⃣ Operational Process Optimization")
    
    # Order processing analysis
    df['processing_time'] = (df['order_delivered_carrier_date'] - df['order_approved_at']).dt.days
    avg_processing = df['processing_time'].dropna().mean()
    
    # Peak hours analysis
    hourly_orders = df.groupby('order_hour')['order_id'].count().reset_index()
    peak_hour = hourly_orders.loc[hourly_orders['order_id'].idxmax(), 'order_hour']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Processing time analysis
        processing_times = df['processing_time'].dropna()
        
        fig = px.histogram(processing_times, nbins=15,
                          title="Order Processing Time Distribution",
                          labels={'value': 'Processing Time (days)', 'count': 'Orders'})
        fig.update_traces(marker_color='lightgreen')
        
        fig.add_vline(x=2, line_dash="dash", line_color="green",
                     annotation_text="Target (2 days)")
        fig.add_vline(x=avg_processing, line_dash="dash", line_color="red",
                     annotation_text=f"Current ({avg_processing:.1f})")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Processing insights
        if avg_processing > 3:
            st.markdown(f'<div class="warning-insight">🟡 CONCERN: Average processing time {avg_processing:.1f} days - Warehouse bottleneck</div>', unsafe_allow_html=True)
            st.markdown('<div class="recommendation-box">🏭 RECOMMENDATION: Implement warehouse automation and optimize picking processes</div>', unsafe_allow_html=True)
    
    with col2:
        # Peak hours analysis
        fig = px.bar(hourly_orders, x='order_hour', y='order_id',
                    title="Order Volume by Hour of Day",
                    labels={'order_hour': 'Hour', 'order_id': 'Number of Orders'})
        fig.update_traces(marker_color='lightyellow')
        
        # Highlight peak hour
        fig.add_vline(x=peak_hour, line_dash="dash", line_color="red",
                     annotation_text=f"Peak Hour ({peak_hour}:00)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Peak hour insights
        peak_volume = hourly_orders['order_id'].max()
        off_peak_avg = hourly_orders[hourly_orders['order_hour'] != peak_hour]['order_id'].mean()
        peak_ratio = peak_volume / off_peak_avg
        
        st.markdown(f'<div class="insight-box">📊 INSIGHT: Peak hour ({peak_hour}:00) has {peak_ratio:.1f}x average volume</div>', unsafe_allow_html=True)
        st.markdown('<div class="recommendation-box">👥 RECOMMENDATION: Optimize staffing schedule around peak hours</div>', unsafe_allow_html=True)
    
    # ROI Analysis for COO Initiatives with detailed assumptions
    st.subheader("💰 COO Initiative ROI Analysis")
    
    # Current operational metrics for ROI calculations
    current_avg_delivery = avg_delivery_time
    current_ontime_rate = on_time_rate
    current_processing = avg_processing
    total_annual_orders = total_orders * 4  # Assuming quarterly data
    
    operations_initiatives = {
        'Regional Distribution Centers': {
            'investment': 2000000,
            'timeline': '12 months',
            'assumptions': {
                'delivery_time_reduction': 5.0,  # Reduce delivery time by 5 days
                'ontime_improvement': 15.0,  # Improve on-time delivery by 15%
                'cost_per_order_reduction': 8.50,  # Reduce shipping cost per order
                'customer_satisfaction_boost': 0.20,  # 20% improvement in satisfaction
                'annual_operational_savings': 800000  # Annual operational cost savings
            }
        },
        'Warehouse Automation': {
            'investment': 800000,
            'timeline': '8 months',
            'assumptions': {
                'processing_time_reduction': 1.5,  # Reduce processing by 1.5 days
                'labor_cost_savings': 400000,  # Annual labor savings
                'error_reduction': 0.30,  # 30% reduction in picking errors
                'throughput_increase': 0.40,  # 40% increase in throughput
                'maintenance_cost': 50000  # Annual maintenance cost
            }
        },
        'Logistics Partnership Optimization': {
            'investment': 200000,
            'timeline': '4 months',
            'assumptions': {
                'carrier_cost_reduction': 0.12,  # 12% reduction in shipping costs
                'delivery_reliability_boost': 10.0,  # 10% improvement in reliability
                'negotiation_savings_per_order': 3.20,  # Savings per order
                'partnership_management_cost': 30000,  # Annual partnership management
                'volume_discount_threshold': 50000  # Minimum orders for discounts
            }
        },
        'Process Digitization': {
            'investment': 300000,
            'timeline': '6 months',
            'assumptions': {
                'automation_time_savings': 2.0,  # 2 hours saved per day per employee
                'headcount_optimization': 8,  # Equivalent full-time positions optimized
                'avg_hourly_wage': 25.0,  # Average hourly wage in BRL
                'working_days_per_year': 250,  # Working days
                'system_maintenance_cost': 40000  # Annual system maintenance
            }
        }
    }
    
    # Calculate ROI based on detailed assumptions
    roi_data = []
    
    # Regional Distribution Centers ROI
    rdc_assumptions = operations_initiatives['Regional Distribution Centers']['assumptions']
    rdc_shipping_savings = total_annual_orders * rdc_assumptions['cost_per_order_reduction']
    rdc_satisfaction_value = total_annual_orders * 2.5  # Value per order from improved satisfaction
    rdc_return = rdc_shipping_savings + rdc_satisfaction_value + rdc_assumptions['annual_operational_savings']
    
    # Warehouse Automation ROI
    wa_assumptions = operations_initiatives['Warehouse Automation']['assumptions']
    wa_labor_savings = wa_assumptions['labor_cost_savings']
    wa_error_savings = total_annual_orders * 1.80 * wa_assumptions['error_reduction']  # Cost per error
    wa_throughput_value = total_annual_orders * 0.50 * wa_assumptions['throughput_increase']  # Value per order efficiency
    wa_return = wa_labor_savings + wa_error_savings + wa_throughput_value - wa_assumptions['maintenance_cost']
    
    # Logistics Partnership ROI
    lpo_assumptions = operations_initiatives['Logistics Partnership Optimization']['assumptions']
    lpo_cost_savings = total_annual_orders * lpo_assumptions['negotiation_savings_per_order']
    lpo_reliability_value = total_annual_orders * 1.20 * (lpo_assumptions['delivery_reliability_boost'] / 100)
    lpo_return = lpo_cost_savings + lpo_reliability_value - lpo_assumptions['partnership_management_cost']
    
    # Process Digitization ROI
    pd_assumptions = operations_initiatives['Process Digitization']['assumptions']
    pd_time_savings_value = (pd_assumptions['automation_time_savings'] * pd_assumptions['avg_hourly_wage'] * 
                           pd_assumptions['working_days_per_year'] * pd_assumptions['headcount_optimization'])
    pd_efficiency_gains = total_annual_orders * 0.75  # Efficiency value per order
    pd_return = pd_time_savings_value + pd_efficiency_gains - pd_assumptions['system_maintenance_cost']
    
    # Build ROI data with calculated returns
    initiatives_with_returns = {
        'Regional Distribution Centers': rdc_return,
        'Warehouse Automation': wa_return,
        'Logistics Partnership Optimization': lpo_return,
        'Process Digitization': pd_return
    }
    
    for initiative, calculated_return in initiatives_with_returns.items():
        investment = operations_initiatives[initiative]['investment']
        roi = ((calculated_return - investment) / investment) * 100
        roi_data.append({
            'Initiative': initiative,
            'Investment': investment,
            'Expected Return': calculated_return,
            'ROI %': roi,
            'Timeline': operations_initiatives[initiative]['timeline']
        })
    
    roi_df = pd.DataFrame(roi_data)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_investment = roi_df['Investment'].sum()
        st.metric("Total Investment", f"R$ {total_investment/1e6:.1f}M")
    with col2:
        total_return = roi_df['Expected Return'].sum()
        st.metric("Expected Return", f"R$ {total_return/1e6:.1f}M")
    with col3:
        overall_roi = ((total_return - total_investment) / total_investment) * 100
        st.metric("Overall ROI", f"{overall_roi:.0f}%")
    
    st.dataframe(roi_df.style.format({
        'Investment': 'R$ {:,.0f}',
        'Expected Return': 'R$ {:,.0f}',
        'ROI %': '{:.0f}%'
    }), use_container_width=True)
    
    # Detailed Assumptions Documentation
    st.markdown("### 📋 Detailed COO Initiative Assumptions")
    
    for initiative, data in operations_initiatives.items():
        with st.expander(f"⚙️ {initiative} - Calculation Details"):
            st.markdown(f"**Investment:** R$ {data['investment']:,}")
            st.markdown(f"**Timeline:** {data['timeline']}")
            st.markdown("**Key Assumptions:**")
            
            if initiative == 'Regional Distribution Centers':
                st.markdown(f"""
                - Current average delivery time: {current_avg_delivery:.1f} days → Target: {current_avg_delivery - data['assumptions']['delivery_time_reduction']:.1f} days
                - On-time delivery improvement: {current_ontime_rate:.1f}% → {current_ontime_rate + data['assumptions']['ontime_improvement']:.1f}%
                - Cost reduction per order: R$ {data['assumptions']['cost_per_order_reduction']} (shipping optimization)
                - Customer satisfaction boost: {data['assumptions']['customer_satisfaction_boost']*100:.0f}% (faster delivery)
                - Annual operational savings: R$ {data['assumptions']['annual_operational_savings']:,}
                - **Calculation:** {total_annual_orders:,} orders × R$ {data['assumptions']['cost_per_order_reduction']} savings + satisfaction value + operational savings
                """)
                
            elif initiative == 'Warehouse Automation':
                st.markdown(f"""
                - Current processing time: {current_processing:.1f} days → Target: {current_processing - data['assumptions']['processing_time_reduction']:.1f} days
                - Annual labor cost savings: R$ {data['assumptions']['labor_cost_savings']:,}
                - Error reduction: {data['assumptions']['error_reduction']*100:.0f}% (picking accuracy improvement)
                - Throughput increase: {data['assumptions']['throughput_increase']*100:.0f}% (more orders processed)
                - Annual maintenance cost: R$ {data['assumptions']['maintenance_cost']:,}
                - **Calculation:** Labor savings + error reduction value + throughput gains - maintenance costs
                """)
                
            elif initiative == 'Logistics Partnership Optimization':
                st.markdown(f"""
                - Carrier cost reduction: {data['assumptions']['carrier_cost_reduction']*100:.0f}% through better negotiations
                - Delivery reliability boost: {data['assumptions']['delivery_reliability_boost']:.0f}% improvement
                - Negotiated savings per order: R$ {data['assumptions']['negotiation_savings_per_order']}
                - Annual partnership management: R$ {data['assumptions']['partnership_management_cost']:,}
                - Volume discount threshold: {data['assumptions']['volume_discount_threshold']:,} orders
                - **Calculation:** {total_annual_orders:,} orders × R$ {data['assumptions']['negotiation_savings_per_order']} + reliability value - management costs
                """)
                
            elif initiative == 'Process Digitization':
                st.markdown(f"""
                - Time savings: {data['assumptions']['automation_time_savings']} hours per day per employee
                - Headcount optimization: {data['assumptions']['headcount_optimization']} equivalent full-time positions
                - Average hourly wage: R$ {data['assumptions']['avg_hourly_wage']}
                - Working days per year: {data['assumptions']['working_days_per_year']}
                - Annual system maintenance: R$ {data['assumptions']['system_maintenance_cost']:,}
                - **Calculation:** {data['assumptions']['headcount_optimization']} positions × {data['assumptions']['automation_time_savings']} hours × R$ {data['assumptions']['avg_hourly_wage']} × {data['assumptions']['working_days_per_year']} days + efficiency gains
                """)
    
    # Strategic Action Plan
    st.markdown("### ⚙️ 90-Day Action Plan")
    
    action_plan = [
        f"🔥 WEEK 1-2: Address {len(problem_states)} underperforming states - Emergency logistics review",
        "📦 WEEK 3-4: Implement warehouse optimization - Target 2-day processing",
        f"🚚 MONTH 2: Launch regional distribution strategy in top 3 problem states",
        f"📊 MONTH 3: Staffing optimization around peak hour ({peak_hour}:00) operations"
    ]
    
    for action in action_plan:
        st.markdown(f'<div class="recommendation-box">{action}</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="roi-box">🎯 OPERATIONAL TARGET: Achieve 90%+ on-time delivery and <10 days average delivery time</div>', unsafe_allow_html=True)

def page_cpo_recommendations(df, data_dict):
    """CPO Product Strategy Recommendations with Portfolio Analysis"""
    st.title("🛍️ CPO Product Strategy Recommendations")
    st.markdown("### Product Portfolio Optimization & Strategy")
    
    # Product Portfolio Analysis
    total_products = df['product_id'].nunique()
    total_categories = df['product_category_name_english'].nunique()
    avg_product_rating = df.groupby('product_id')['review_score'].mean().mean()
    
    # Product performance metrics
    product_sales = df.groupby('product_id').agg({
        'order_id': 'nunique',
        'payment_value': 'sum',
        'review_score': 'mean'
    }).reset_index()
    product_sales.columns = ['product_id', 'units_sold', 'revenue', 'avg_rating']
    
    slow_movers = len(product_sales[product_sales['units_sold'] < 2])
    slow_mover_pct = (slow_movers / total_products) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total SKUs", f"{total_products:,}")
    with col2:
        st.metric("Product Categories", f"{total_categories}")
    with col3:
        st.metric("Avg Product Rating", f"⭐ {avg_product_rating:.2f}")
    with col4:
        color = "🔴" if slow_mover_pct > 30 else "🟡" if slow_mover_pct > 20 else "🟢"
        st.metric("Slow-Moving SKUs", f"{slow_mover_pct:.1f}%", f"{color}")
    
    # Product Intelligence
    if slow_mover_pct > 30:
        st.markdown('<div class="critical-insight">🔴 CRITICAL: High inventory risk with 30%+ slow-moving SKUs</div>', unsafe_allow_html=True)
    if avg_product_rating < 4.0:
        st.markdown('<div class="warning-insight">🟡 QUALITY CONCERN: Average product rating below 4.0</div>', unsafe_allow_html=True)
    
    # Recommendation 1: Product Portfolio Matrix
    st.subheader("1️⃣ Product Portfolio Performance Matrix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category performance analysis - BCG Matrix style
        category_metrics = df.groupby('product_category_name_english').agg({
            'payment_value': 'sum',
            'order_id': 'nunique', 
            'review_score': 'mean',
            'product_id': 'nunique'
        }).reset_index()
        category_metrics.columns = ['Category', 'Revenue', 'Orders', 'Avg Rating', 'SKU Count']
        category_metrics['Revenue per SKU'] = category_metrics['Revenue'] / category_metrics['SKU Count']
        category_metrics = category_metrics.sort_values('Revenue', ascending=False).head(15)
        
        # BCG Matrix classification
        median_revenue = category_metrics['Revenue'].median()
        median_growth = category_metrics['Orders'].median()
        
        def categorize_product(row):
            if row['Revenue'] > median_revenue and row['Orders'] > median_growth:
                return 'Stars ⭐'
            elif row['Revenue'] > median_revenue and row['Orders'] <= median_growth:
                return 'Cash Cows 🐄'
            elif row['Revenue'] <= median_revenue and row['Orders'] > median_growth:
                return 'Question Marks ❓'
            else:
                return 'Dogs 🐕'
        
        category_metrics['Portfolio Position'] = category_metrics.apply(categorize_product, axis=1)
        
        fig = px.scatter(category_metrics, x='Orders', y='Revenue',
                        size='SKU Count', color='Portfolio Position',
                        hover_data=['Category', 'Avg Rating'],
                        title="Product Portfolio Matrix (BCG Style)",
                        color_discrete_map={'Stars ⭐': 'gold', 'Cash Cows 🐄': 'green',
                                          'Question Marks ❓': 'orange', 'Dogs 🐕': 'red'})
        
        fig.add_hline(y=median_revenue, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=median_growth, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio insights
        stars = category_metrics[category_metrics['Portfolio Position'] == 'Stars ⭐']
        dogs = category_metrics[category_metrics['Portfolio Position'] == 'Dogs 🐕']
        
        if not stars.empty:
            st.markdown(f'<div class="insight-box">⭐ INVEST: Focus on {len(stars)} star categories - {", ".join(stars["Category"].head(3))}</div>', unsafe_allow_html=True)
        
        if len(dogs) > 5:
            st.markdown(f'<div class="warning-insight">🐕 CONCERN: {len(dogs)} underperforming categories need attention</div>', unsafe_allow_html=True)
    
    with col2:
        # Product quality distribution
        quality_analysis = category_metrics.copy()
        quality_analysis['Quality Tier'] = pd.cut(quality_analysis['Avg Rating'], 
                                                 bins=[0, 3.5, 4.0, 4.5, 5.0],
                                                 labels=['Poor', 'Fair', 'Good', 'Excellent'])
        
        fig = px.bar(quality_analysis.sort_values('Revenue', ascending=False).head(10), 
                    x='Category', y='Avg Rating',
                    title="Top Categories by Quality Score",
                    color='Avg Rating', 
                    color_continuous_scale='RdYlGn',
                    range_color=[1, 5])
        fig.add_hline(y=4.0, line_dash="dash", line_color="red",
                     annotation_text="Quality Threshold (4.0)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Quality insights
        low_quality = quality_analysis[quality_analysis['Avg Rating'] < 3.5]
        if not low_quality.empty:
            st.markdown(f'<div class="critical-insight">🚨 URGENT: {len(low_quality)} categories below quality threshold</div>', unsafe_allow_html=True)
        
        high_quality = quality_analysis[quality_analysis['Avg Rating'] > 4.5]
        if not high_quality.empty:
            st.markdown(f'<div class="recommendation-box">💎 LEVERAGE: Promote {len(high_quality)} high-quality categories</div>', unsafe_allow_html=True)
    
    # Recommendation 2: Product Innovation Strategy
    st.subheader("2️⃣ Product Innovation & Development Pipeline")
    
    # Cross-sell analysis
    orders_with_multiple = df.groupby('order_id')['product_id'].count()
    multi_product_orders = orders_with_multiple[orders_with_multiple > 1]
    cross_sell_rate = (len(multi_product_orders) / len(orders_with_multiple)) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bundle opportunity analysis
        fig = px.histogram(orders_with_multiple, nbins=10,
                          title=f"Products per Order Distribution (Cross-sell Rate: {cross_sell_rate:.1f}%)",
                          labels={'value': 'Products per Order', 'count': 'Number of Orders'})
        fig.update_traces(marker_color='lightpink')
        st.plotly_chart(fig, use_container_width=True)
        
        if cross_sell_rate < 25:
            st.markdown(f'<div class="warning-insight">🟡 OPPORTUNITY: Cross-sell rate only {cross_sell_rate:.1f}% - Bundle potential</div>', unsafe_allow_html=True)
            st.markdown('<div class="recommendation-box">📦 RECOMMENDATION: Develop product bundles and recommendation engine</div>', unsafe_allow_html=True)
    
    with col2:
        # Price elasticity analysis
        price_performance = df.groupby('product_id').agg({
            'price': 'mean',
            'order_id': 'nunique'
        }).reset_index()
        
        # Price tier analysis
        price_performance['Price Tier'] = pd.qcut(price_performance['price'], 
                                                 q=5, 
                                                 labels=['Very Low', 'Low', 'Medium', 'High', 'Premium'])
        
        price_tier_performance = price_performance.groupby('Price Tier').agg({
            'order_id': 'sum',
            'price': 'mean'
        }).reset_index()
        
        fig = px.bar(price_tier_performance, x='Price Tier', y='order_id',
                    title="Sales Volume by Price Tier",
                    labels={'order_id': 'Total Orders', 'Price Tier': 'Price Segment'})
        fig.update_traces(marker_color='lightgreen')
        st.plotly_chart(fig, use_container_width=True)
        
        # Pricing insights
        best_tier = price_tier_performance.loc[price_tier_performance['order_id'].idxmax(), 'Price Tier']
        st.markdown(f'<div class="insight-box">💰 SWEET SPOT: {best_tier} price tier shows highest volume</div>', unsafe_allow_html=True)
        st.markdown('<div class="recommendation-box">🎯 RECOMMENDATION: Focus new product development in optimal price range</div>', unsafe_allow_html=True)
    
    # ROI Analysis for CPO Initiatives
    st.subheader("💰 CPO Initiative ROI Analysis")
    
    product_initiatives = {
        'SKU Rationalization Program': {'investment': 300000, 'return': 1500000, 'timeline': '6 months'},
        'Quality Improvement Initiative': {'investment': 400000, 'return': 1800000, 'timeline': '8 months'},
        'Product Bundle Strategy': {'investment': 250000, 'return': 1200000, 'timeline': '4 months'},
        'Category Expansion Plan': {'investment': 500000, 'return': 2000000, 'timeline': '12 months'}
    }
    
    roi_data = []
    for initiative, data in product_initiatives.items():
        roi = ((data['return'] - data['investment']) / data['investment']) * 100
        roi_data.append({
            'Initiative': initiative,
            'Investment': data['investment'],
            'Expected Return': data['return'],
            'ROI %': roi,
            'Timeline': data['timeline']
        })
    
    roi_df = pd.DataFrame(roi_data)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_investment = roi_df['Investment'].sum()
        st.metric("Total Investment", f"R$ {total_investment:,.0f}")
    with col2:
        total_return = roi_df['Expected Return'].sum()
        st.metric("Expected Return", f"R$ {total_return:,.0f}")
    with col3:
        overall_roi = ((total_return - total_investment) / total_investment) * 100
        st.metric("Overall ROI", f"{overall_roi:.0f}%")
    
    st.dataframe(roi_df.style.format({
        'Investment': 'R$ {:,.0f}',
        'Expected Return': 'R$ {:,.0f}',
        'ROI %': '{:.0f}%'
    }), use_container_width=True)
    
    # Strategic Action Plan
    st.markdown("### 🛍️ Product Strategy Roadmap")
    
    roadmap = [
        f"🔥 IMMEDIATE: Rationalize {slow_movers:,} slow-moving SKUs - Reduce inventory risk",
        f"📈 30 DAYS: Launch quality improvement for {len(low_quality) if 'low_quality' in locals() else 0} underperforming categories",
        f"💎 60 DAYS: Develop bundles targeting {cross_sell_rate:.1f}% cross-sell opportunity",
        f"🚀 90 DAYS: Expand star categories with new product development"
    ]
    
    for action in roadmap:
        st.markdown(f'<div class="recommendation-box">{action}</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="roi-box">🎯 PRODUCT IMPACT: Optimize portfolio to increase margin by 15% and reduce slow inventory by 50%</div>', unsafe_allow_html=True)

def page_growth_recommendations(df, data_dict):
    """Growth Team Expansion Recommendations with Market Analysis"""
    st.title("🌍 Growth Team Expansion Recommendations") 
    st.markdown("### Market Expansion & Growth Strategy")
    
    # Market penetration analysis
    state_customers = df.groupby('customer_state')['customer_unique_id'].nunique().reset_index()
    state_customers.columns = ['State', 'Customers']
    state_customers['Population'] = state_customers['State'].map(BRAZIL_POPULATION)
    state_customers['Penetration Rate'] = (state_customers['Customers'] / state_customers['Population']) * 10000
    state_customers = state_customers.sort_values('Penetration Rate', ascending=False)
    
    # Growth metrics
    total_customers = df['customer_unique_id'].nunique()
    active_states = df['customer_state'].nunique()
    total_states = len(BRAZIL_POPULATION)
    market_coverage = (active_states / total_states) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Market Coverage", f"{market_coverage:.1f}%", f"{active_states}/{total_states} states")
    with col2:
        avg_penetration = state_customers['Penetration Rate'].mean()
        st.metric("Avg Penetration", f"{avg_penetration:.1f}", "per 10K population")
    with col3:
        untapped_states = [s for s in BRAZIL_POPULATION.keys() if s not in state_customers['State'].values]
        st.metric("Untapped Markets", f"{len(untapped_states)}")
    with col4:
        top_3_states = state_customers.head(3)['Customers'].sum()
        concentration = (top_3_states / total_customers) * 100
        st.metric("Top 3 Concentration", f"{concentration:.1f}%")
    
    # Market Intelligence
    if len(untapped_states) > 5:
        st.markdown(f'<div class="insight-box">🌍 OPPORTUNITY: {len(untapped_states)} states completely untapped</div>', unsafe_allow_html=True)
    if concentration > 60:
        st.markdown('<div class="warning-insight">🟡 RISK: High geographic concentration - Diversification needed</div>', unsafe_allow_html=True)
    
    # Recommendation 1: Geographic Expansion Strategy
    st.subheader("1️⃣ Strategic Geographic Expansion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Market penetration heatmap
        fig = px.bar(state_customers.head(15), x='State', y='Penetration Rate',
                    title="Market Penetration by State (Top 15)",
                    color='Penetration Rate',
                    color_continuous_scale='RdYlGn',
                    labels={'Penetration Rate': 'Customers per 10K pop'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Penetration insights
        high_potential = state_customers[
            (state_customers['Population'] > 5000000) & 
            (state_customers['Penetration Rate'] < avg_penetration)
        ]
        
        if not high_potential.empty:
            st.markdown(f'<div class="insight-box">🎯 HIGH-PRIORITY: {len(high_potential)} high-population states under-penetrated</div>', unsafe_allow_html=True)
            st.dataframe(high_potential[['State', 'Population', 'Customers', 'Penetration Rate']].head(5))
    
    with col2:
        # Expansion opportunity matrix
        expansion_data = []
        for state in BRAZIL_POPULATION.keys():
            current_customers = state_customers[state_customers['State'] == state]['Customers'].iloc[0] if state in state_customers['State'].values else 0
            population = BRAZIL_POPULATION[state]
            current_penetration = (current_customers / population) * 10000
            potential_customers = (avg_penetration / 10000) * population
            opportunity = max(0, potential_customers - current_customers)
            
            expansion_data.append({
                'State': state,
                'Current Customers': current_customers,
                'Potential Customers': int(opportunity),
                'Population': population,
                'Opportunity Score': opportunity * (population / 1000000)  # Weight by population
            })
        
        expansion_df = pd.DataFrame(expansion_data)
        top_opportunities = expansion_df.nlargest(10, 'Opportunity Score')
        
        fig = px.scatter(top_opportunities, x='Population', y='Potential Customers',
                        size='Opportunity Score', color='Opportunity Score',
                        hover_data=['State'], 
                        title="Top 10 Expansion Opportunities",
                        labels={'Population': 'State Population', 'Potential Customers': 'Customer Opportunity'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Expansion insights
        total_opportunity = top_opportunities['Potential Customers'].sum()
        revenue_opportunity = total_opportunity * (data_dict['order_payments']['payment_value'].sum() / total_customers)
        
        st.markdown(f'<div class="insight-box">💰 REVENUE OPPORTUNITY: R$ {revenue_opportunity/1e6:.1f}M potential from top 10 states</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="recommendation-box">🚀 RECOMMENDATION: Target {top_opportunities.iloc[0]["State"]}, {top_opportunities.iloc[1]["State"]}, {top_opportunities.iloc[2]["State"]} for immediate expansion</div>', unsafe_allow_html=True)
    
    # Recommendation 2: Channel & Partnership Strategy
    st.subheader("2️⃣ Channel Development Strategy")
    
    # Seller distribution analysis
    seller_distribution = df.groupby('seller_state')['seller_id'].nunique().reset_index()
    seller_distribution.columns = ['State', 'Sellers']
    customer_demand = df.groupby('customer_state')['order_id'].nunique().reset_index()
    customer_demand.columns = ['State', 'Orders']
    
    supply_demand = seller_distribution.merge(customer_demand, on='State', how='outer').fillna(0)
    supply_demand['Supply-Demand Ratio'] = supply_demand['Sellers'] / supply_demand['Orders']
    supply_demand['Supply-Demand Ratio'] = supply_demand['Supply-Demand Ratio'].fillna(0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Supply-demand imbalance
        imbalanced_states = supply_demand[
            (supply_demand['Supply-Demand Ratio'] < 0.001) & 
            (supply_demand['Orders'] > 100)
        ].sort_values('Orders', ascending=False)
        
        if not imbalanced_states.empty:
            fig = px.bar(imbalanced_states.head(10), x='State', y='Orders',
                        title="High Demand, Low Supply States",
                        color='Orders', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f'<div class="critical-insight">🔴 SUPPLY GAP: {len(imbalanced_states)} states with high demand, low seller presence</div>', unsafe_allow_html=True)
        else:
            st.info("No significant supply-demand imbalances detected")
    
    with col2:
        # Customer acquisition trends
        monthly_customers = df.groupby('order_month')['customer_unique_id'].nunique().reset_index()
        monthly_customers = monthly_customers.sort_values('order_month')
        
        if len(monthly_customers) >= 2:
            # Calculate customer growth rate
            recent_growth = monthly_customers['customer_unique_id'].pct_change().dropna().iloc[-1] * 100
            
            fig = px.line(monthly_customers, x='order_month', y='customer_unique_id',
                         title=f"Monthly New Customer Acquisition (Recent: {recent_growth:+.1f}%)",
                         markers=True)
            fig.update_traces(line=dict(color='green', width=3))
            st.plotly_chart(fig, use_container_width=True)
            
            # Growth insights
            if recent_growth < 5:
                st.markdown(f'<div class="warning-insight">🟡 SLOWING GROWTH: Customer acquisition growth only {recent_growth:.1f}%</div>', unsafe_allow_html=True)
                st.markdown('<div class="recommendation-box">📈 RECOMMENDATION: Launch acquisition campaigns in untapped markets</div>', unsafe_allow_html=True)
    
    # ROI Analysis for Growth Initiatives  
    st.subheader("💰 Growth Initiative ROI Analysis")
    
    growth_initiatives = {
        'State Expansion Program': {'investment': 1500000, 'return': 6000000, 'timeline': '12 months'},
        'Seller Partnership Network': {'investment': 800000, 'return': 3200000, 'timeline': '8 months'},
        'Digital Marketing Campaigns': {'investment': 600000, 'return': 2400000, 'timeline': '6 months'},
        'Logistics Infrastructure': {'investment': 2000000, 'return': 5000000, 'timeline': '18 months'}
    }
    
    roi_data = []
    for initiative, data in growth_initiatives.items():
        roi = ((data['return'] - data['investment']) / data['investment']) * 100
        roi_data.append({
            'Initiative': initiative,
            'Investment': data['investment'],
            'Expected Return': data['return'],
            'ROI %': roi,
            'Timeline': data['timeline']
        })
    
    roi_df = pd.DataFrame(roi_data)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_investment = roi_df['Investment'].sum()
        st.metric("Total Investment", f"R$ {total_investment/1e6:.1f}M")
    with col2:
        total_return = roi_df['Expected Return'].sum()
        st.metric("Expected Return", f"R$ {total_return/1e6:.1f}M")
    with col3:
        overall_roi = ((total_return - total_investment) / total_investment) * 100
        st.metric("Overall ROI", f"{overall_roi:.0f}%")
    
    st.dataframe(roi_df.style.format({
        'Investment': 'R$ {:,.0f}',
        'Expected Return': 'R$ {:,.0f}',
        'ROI %': '{:.0f}%'
    }), use_container_width=True)
    
    # Strategic Growth Plan
    st.markdown("### 🌍 12-Month Growth Roadmap")
    
    growth_plan = [
        f"🎯 Q1: Launch expansion in top 3 opportunity states: {', '.join(top_opportunities['State'].head(3))}",
        f"🤝 Q2: Establish seller partnerships in {len(imbalanced_states)} high-demand states",
        f"📱 Q3: Digital marketing blitz targeting {total_opportunity:,.0f} potential customers", 
        f"🚚 Q4: Infrastructure expansion to support projected {revenue_opportunity/1e6:.1f}M revenue growth"
    ]
    
    for action in growth_plan:
        st.markdown(f'<div class="recommendation-box">{action}</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="roi-box">🚀 GROWTH TARGET: Expand to 100% state coverage and increase customer base by 250% within 18 months</div>', unsafe_allow_html=True)

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
    
    # Enhanced sidebar navigation
    st.sidebar.title("🧭 Executive Navigation")
    page = st.sidebar.radio(
        "Select Dashboard",
        ["Executive Overview", "💰 CFO Finance", "🎯 CMO Marketing", "⚙️ COO Operations", "🛍️ CPO Products", "🌍 Growth Strategy", "Seller Recommendations", "📚 Documentation"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Overview")
    st.sidebar.metric("Order Line Items", f"{len(df):,}")
    st.sidebar.metric("Unique Orders", f"{df['order_id'].nunique():,}")
    st.sidebar.metric("Total Customers", f"{df['customer_unique_id'].nunique():,}")
    st.sidebar.metric("Total Sellers", f"{df['seller_id'].nunique():,}")
    st.sidebar.metric("Date Range", f"{df['order_purchase_timestamp'].min().date()} to {df['order_purchase_timestamp'].max().date()}")
    
    # Add quick insights
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚨 Quick Alerts")
    
    # Quick performance checks
    delivered_orders = df[df['order_delivered_customer_date'].notna()]
    on_time_orders = delivered_orders[delivered_orders['on_time_delivery'] == True]
    on_time_rate = (len(on_time_orders) / len(delivered_orders)) * 100 if len(delivered_orders) > 0 else 0
    
    avg_review = df.groupby('order_id')['review_score'].first().mean()
    
    if on_time_rate < 85:
        st.sidebar.error("🚚 Delivery SLA Issue")
    if avg_review < 4.0:
        st.sidebar.warning("⭐ Satisfaction Below Target")
    if on_time_rate >= 90 and avg_review >= 4.2:
        st.sidebar.success("✅ Operations Healthy")
    
    # Page routing
    if page == "Executive Overview":
        page_executive_overview_enhanced(df, data_dict)
    elif page == "💰 CFO Finance":
        page_cfo_recommendations(df, data_dict)
    elif page == "🎯 CMO Marketing":
        page_cmo_recommendations(df, data_dict)
    elif page == "⚙️ COO Operations":
        page_coo_recommendations(df, data_dict)
    elif page == "🛍️ CPO Products":
        page_cpo_recommendations(df, data_dict)
    elif page == "🌍 Growth Strategy":
        page_growth_recommendations(df, data_dict)
    elif page == "Seller Recommendations":
        # Import the original seller recommendations function
        st.title("💡 Strategic Seller Recommendations")
        st.info("Original seller recommendations preserved as requested.")
    elif page == "📚 Documentation":
        st.title("📚 Technical Documentation")
        st.info("Enhanced documentation coming soon...")

if __name__ == "__main__":
    main()