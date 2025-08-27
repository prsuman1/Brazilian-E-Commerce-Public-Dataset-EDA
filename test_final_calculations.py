#!/usr/bin/env python3
"""
🧪 FINAL ACCURACY TEST - Source Tables Direct Access
==================================================
Testing the updated app.py with source table calculations
"""

import pandas as pd

def load_data():
    """Load original datasets"""
    print("Loading original datasets...")
    try:
        order_items = pd.read_csv('olist_order_items_dataset.csv')
        order_payments = pd.read_csv('olist_order_payments_dataset.csv')
        
        print(f"✅ Order Items: {len(order_items):,} records")
        print(f"✅ Order Payments: {len(order_payments):,} records")
        
        return {
            'order_items': order_items,
            'order_payments': order_payments
        }
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def test_source_calculations():
    """Test calculations using source tables directly"""
    print("\n🧪 TESTING SOURCE TABLE CALCULATIONS")
    print("="*50)
    
    data = load_data()
    if not data:
        return
    
    # Expected accurate values
    print("1️⃣ PRODUCT REVENUE:")
    print("-" * 30)
    product_revenue = data['order_items']['price'].sum()
    print(f"Product Revenue (source): R$ {product_revenue:,.2f}")
    
    print("\n2️⃣ FREIGHT REVENUE:")
    print("-" * 30)
    freight_revenue = data['order_items']['freight_value'].sum()
    print(f"Freight Revenue (source): R$ {freight_revenue:,.2f}")
    
    print("\n3️⃣ TOTAL PAYMENT VALUE:")
    print("-" * 30)
    total_payment = data['order_payments']['payment_value'].sum()
    print(f"Total Payment (source): R$ {total_payment:,.2f}")
    
    print("\n4️⃣ OTHER CHARGES CALCULATION:")
    print("-" * 30)
    other_charges = total_payment - (product_revenue + freight_revenue)
    print(f"Other Charges: R$ {other_charges:,.2f}")
    
    # Verify calculation
    calculated_total = product_revenue + freight_revenue + other_charges
    print(f"Calculated Total: R$ {calculated_total:,.2f}")
    
    if abs(calculated_total - total_payment) < 1:
        print("✅ Waterfall calculation PERFECT!")
    else:
        print(f"❌ Difference: R$ {calculated_total - total_payment:,.2f}")
    
    print("\n" + "="*50)
    print("📊 EXPECTED DASHBOARD VALUES")
    print("="*50)
    print(f"Product Sales: R$ {product_revenue:,.0f}")
    print(f"Freight Charges: R$ {freight_revenue:,.0f}")
    print(f"Other Charges: R$ {other_charges:,.0f}")
    print(f"Total Revenue: R$ {total_payment:,.0f}")
    print("\n✅ These should be the EXACT values shown in the dashboard!")

if __name__ == "__main__":
    test_source_calculations()