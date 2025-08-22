import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

# Association Rule Mining
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Survival Analysis
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullFitter
from lifelines.utils import concordance_index

# Ranking
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score
import joblib


class AIFamilyRecommendationEngine:
    def __init__(self):
        # Core models
        self.association_rules_model = None
        self.price_predictor = None
        self.survival_model = None
        self.ranking_model = None
        
        # Data preprocessing components
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Product catalog and data storage
        self.product_catalog = {}  # SKU -> product details
        self.category_products = {}  # category -> brand -> list of products
        self.price_bands = {}  # category -> brand -> price statistics
        self.reference_data = None  # Will store processed transaction data
        
        # Brand ecosystem mapping for ecosystem completion
        self.brand_ecosystems = {
            'samsung': ['phones', 'tablets', 'watches', 'audio', 'tvs', 'laptops', 'appliances'],
            'apple': ['phones', 'tablets', 'watches', 'audio', 'laptops', 'tv_streaming', 'accessories'],
            'google': ['phones', 'tablets', 'audio', 'smart_home', 'streaming', 'nest'],
            'amazon': ['tablets', 'audio', 'smart_home', 'streaming', 'alexa_devices'],
            'microsoft': ['laptops', 'tablets', 'gaming', 'productivity', 'surface'],
            'sony': ['audio', 'gaming', 'tvs', 'cameras', 'entertainment'],
            'lg': ['tvs', 'appliances', 'smartphones', 'audio'],
            'whirlpool': ['appliances', 'refrigerator', 'microwave', 'washer', 'dryer'],
            'ge': ['appliances', 'refrigerator', 'microwave', 'dishwasher']
        }
        
    def load_and_preprocess_data(self, catalog_path, lineitem_path, orderdetail_path):
        """Load and preprocess all data files"""
        print("📊 Loading and preprocessing data...")
        
        try:
            # Load catalog data
            catalog_df = pd.read_csv(catalog_path)
            catalog_df['data'] = catalog_df['data'].apply(json.loads)
            
            # Extract product information
            for _, row in catalog_df.iterrows():
                data = row['data']
                product_name = data.get('product_display_name', f"Product {row['sku']}")
                
                # Detect brand and normalize category
                brand = self.detect_brand(product_name, row['sku'])
                raw_category = data.get('taxonomy', {}).get('product_category', 'unknown').lower()
                normalized_category = self.normalize_category(raw_category)
                
                self.product_catalog[row['sku']] = {
                    'sku': row['sku'],
                    'product_name': product_name,
                    'product_category': normalized_category,
                    'product_family': data.get('taxonomy', {}).get('product_family', 'unknown'),
                    'brand': brand
                }
                
                # Group by category and brand
                if normalized_category not in self.category_products:
                    self.category_products[normalized_category] = {}
                if brand not in self.category_products[normalized_category]:
                    self.category_products[normalized_category][brand] = []
                self.category_products[normalized_category][brand].append(self.product_catalog[row['sku']])
            
            print(f"✅ Loaded {len(self.product_catalog)} products from catalog")
            
            # Load line item data (pricing)
            lineitem_df = pd.read_csv(lineitem_path)
            lineitem_df['data'] = lineitem_df['data'].apply(json.loads)
            
            pricing_data = {}
            for _, row in lineitem_df.iterrows():
                data = row['data']
                sku = data.get('sku', '')
                price = data.get('sale_price', {}).get('value', 0)
                currency = data.get('sale_price', {}).get('currency', 'USD')
                
                if sku and price > 0:
                    pricing_data[sku] = {'price': price, 'currency': currency}
            
            print(f"✅ Loaded pricing for {len(pricing_data)} products")
            
            # Load order details
            orders_df = pd.read_csv(orderdetail_path)
            orders_df['data'] = orders_df['data'].apply(json.loads)
            
            # Process orders
            processed_orders = []
            for _, order in orders_df.iterrows():
                order_data = order['data']
                user_id = order_data.get('user_info', {}).get('identity_id', '')
                family_id = order_data.get('user_info', {}).get('family_id', user_id)  # Use user_id if no family_id
                submission_date = order_data.get('submission_date', '')
                total_cost = order_data.get('cost', {}).get('total', 0)
                
                line_items = order_data.get('line_items', {})
                for item_id, item_data in line_items.items():
                    sku = item_data.get('sku', '')
                    item_cost = item_data.get('line_item_cost', {}).get('total', 0)
                    
                    # Calculate quantity from nested line items
                    nested_items = item_data.get('line_items', {})
                    quantity = len(nested_items) if nested_items else 1
                    
                    # Get product info and pricing
                    if sku in self.product_catalog:
                        product_info = self.product_catalog[sku]
                        price = pricing_data.get(sku, {}).get('price', item_cost)
                        
                        processed_orders.append({
                            'user_id': user_id,
                            'family_id': family_id,
                            'sku': sku,
                            'product_name': product_info['product_name'],
                            'product_category': product_info['product_category'],
                            'product_family': product_info['product_family'],
                            'brand': product_info['brand'],
                            'purchase_date': pd.to_datetime(submission_date),
                            'price': price,
                            'item_cost': item_cost,
                            'quantity': quantity,
                            'total_order_cost': total_cost
                        })
            
            # Convert to DataFrame
            final_df = pd.DataFrame(processed_orders)
            
            if len(final_df) == 0:
                print("❌ No valid order data found. Creating sample data...")
                final_df = self.create_sample_data()
            else:
                # Calculate days since purchase
                final_df['days_since_purchase'] = (datetime.now() - final_df['purchase_date'].dt.tz_localize(None)).dt.days
                
                # Filter out invalid data
                final_df = final_df[
                    (final_df['price'] > 0) &
                    (final_df['product_category'] != 'unknown') &
                    (~final_df['sku'].str.startswith('DM_', na=False))
                ]
            
            print(f"✅ Processed {len(final_df)} order items for {final_df['family_id'].nunique()} families")
            
            return final_df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("📝 Creating sample data for demonstration...")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create comprehensive sample data for demonstration"""
        print("🎯 Creating enhanced sample data...")
        
        # Sample families with different patterns
        families_data = []
        base_date = datetime.now() - timedelta(days=365*2)
        
        # Family 1: Samsung Ecosystem Enthusiast
        family_purchases = [
            {'family_id': 'family_samsung_tech', 'sku': 'SM-G991U', 'name': 'Galaxy S21 5G 128GB', 
             'category': 'phones', 'brand': 'samsung', 'price': 699, 'days_ago': 60},
            {'family_id': 'family_samsung_tech', 'sku': 'SM-R630N', 'name': 'Galaxy Watch4 Classic 46mm', 
             'category': 'watches', 'brand': 'samsung', 'price': 379, 'days_ago': 120},
            {'family_id': 'family_samsung_tech', 'sku': 'SM-R180N', 'name': 'Galaxy Buds2', 
             'category': 'audio', 'brand': 'samsung', 'price': 149, 'days_ago': 45},
        ]
        
        # Family 2: Apple Ecosystem
        family_purchases.extend([
            {'family_id': 'family_apple_lovers', 'sku': 'IPHONE13-128', 'name': 'iPhone 13 128GB', 
             'category': 'phones', 'brand': 'apple', 'price': 799, 'days_ago': 90},
            {'family_id': 'family_apple_lovers', 'sku': 'APPLEWATCH8', 'name': 'Apple Watch Series 8 45mm', 
             'category': 'watches', 'brand': 'apple', 'price': 429, 'days_ago': 150},
            {'family_id': 'family_apple_lovers', 'sku': 'AIRPODSPRO2', 'name': 'AirPods Pro 2nd Generation', 
             'category': 'audio', 'brand': 'apple', 'price': 249, 'days_ago': 30},
        ])
        
        # Family 3: Mixed Brands - High Spender
        family_purchases.extend([
            {'family_id': 'family_mixed_premium', 'sku': 'PIXEL7-128', 'name': 'Google Pixel 7 128GB', 
             'category': 'phones', 'brand': 'google', 'price': 599, 'days_ago': 40},
            {'family_id': 'family_mixed_premium', 'sku': 'SONY-WH1000XM4', 'name': 'Sony WH-1000XM4 Headphones', 
             'category': 'audio', 'brand': 'sony', 'price': 349, 'days_ago': 80},
            {'family_id': 'family_mixed_premium', 'sku': 'LG-OLED65C1', 'name': '65" OLED C1 Series 4K TV', 
             'category': 'tvs', 'brand': 'lg', 'price': 1299, 'days_ago': 300},
            {'family_id': 'family_mixed_premium', 'sku': 'WHIRLPOOL-WRF535', 'name': 'Whirlpool French Door Refrigerator', 
             'category': 'appliances', 'brand': 'whirlpool', 'price': 1599, 'days_ago': 400},
        ])
        
        # Family 4: Budget Conscious
        family_purchases.extend([
            {'family_id': 'family_budget_smart', 'sku': 'IPHONESE3-64', 'name': 'iPhone SE 3rd Gen 64GB', 
             'category': 'phones', 'brand': 'apple', 'price': 429, 'days_ago': 25},
            {'family_id': 'family_budget_smart', 'sku': 'SM-T500N', 'name': 'Galaxy Tab A7 32GB', 
             'category': 'tablets', 'brand': 'samsung', 'price': 229, 'days_ago': 120},
        ])
        
        # Family 5: Tech Upgrader (old devices needing upgrade)
        family_purchases.extend([
            {'family_id': 'family_tech_upgrader', 'sku': 'IPHONE11-64', 'name': 'iPhone 11 64GB', 
             'category': 'phones', 'brand': 'apple', 'price': 499, 'days_ago': 900},  # ~2.5 years old
            {'family_id': 'family_tech_upgrader', 'sku': 'SM-T870N', 'name': 'Galaxy Tab S7 128GB', 
             'category': 'tablets', 'brand': 'samsung', 'price': 629, 'days_ago': 800},  # ~2.2 years old
            {'family_id': 'family_tech_upgrader', 'sku': 'APPLEWATCH6', 'name': 'Apple Watch Series 6 44mm', 
             'category': 'watches', 'brand': 'apple', 'price': 399, 'days_ago': 850},  # ~2.3 years old
        ])
        
        # Convert to structured data
        sample_data = []
        for purchase in family_purchases:
            purchase_date = base_date + timedelta(days=730 - purchase['days_ago'])
            
            # Add to product catalog
            self.product_catalog[purchase['sku']] = {
                'sku': purchase['sku'],
                'product_name': purchase['name'],
                'product_category': purchase['category'],
                'product_family': purchase['category'],
                'brand': purchase['brand']
            }
            
            # Add to category products
            if purchase['category'] not in self.category_products:
                self.category_products[purchase['category']] = {}
            if purchase['brand'] not in self.category_products[purchase['category']]:
                self.category_products[purchase['category']][purchase['brand']] = []
            self.category_products[purchase['category']][purchase['brand']].append(self.product_catalog[purchase['sku']])
            
            sample_data.append({
                'family_id': purchase['family_id'],
                'user_id': purchase['family_id'] + '_user1',
                'sku': purchase['sku'],
                'product_name': purchase['name'],
                'product_category': purchase['category'],
                'product_family': purchase['category'],
                'brand': purchase['brand'],
                'purchase_date': purchase_date,
                'price': purchase['price'],
                'item_cost': purchase['price'],
                'quantity': 1,
                'total_order_cost': purchase['price'],
                'days_since_purchase': purchase['days_ago']
            })
        
        # Add additional products to catalog for recommendations
        additional_products = [
            # Samsung products
            {'sku': 'SM-T870NZKAXAR', 'name': 'Galaxy Tab S7 128GB', 'category': 'tablets', 'brand': 'samsung', 'price': 629},
            {'sku': 'QN65Q60T', 'name': '65" QLED 4K Q60T Smart TV', 'category': 'tvs', 'brand': 'samsung', 'price': 899},
            {'sku': 'NP950XDB', 'name': 'Galaxy Book Pro 15"', 'category': 'laptops', 'brand': 'samsung', 'price': 1199},
            {'sku': 'RF23M8070', 'name': 'Samsung French Door Refrigerator', 'category': 'appliances', 'brand': 'samsung', 'price': 1899},
            
            # Apple products
            {'sku': 'IPADAIR5-64', 'name': 'iPad Air 5th Gen 64GB', 'category': 'tablets', 'brand': 'apple', 'price': 599},
            {'sku': 'MBA13-M2-256', 'name': 'MacBook Air 13" M2 256GB', 'category': 'laptops', 'brand': 'apple', 'price': 1199},
            {'sku': 'APPLETV4K', 'name': 'Apple TV 4K 128GB', 'category': 'tv_streaming', 'brand': 'apple', 'price': 179},
            
            # Other brands
            {'sku': 'PIXEL-TABLET', 'name': 'Google Pixel Tablet 128GB', 'category': 'tablets', 'brand': 'google', 'price': 399},
            {'sku': 'WHIRLPOOL-WMC30516', 'name': 'Whirlpool Countertop Microwave', 'category': 'appliances', 'brand': 'whirlpool', 'price': 149},
            {'sku': 'SONY-PS5', 'name': 'PlayStation 5 Console', 'category': 'gaming', 'brand': 'sony', 'price': 499},
            {'sku': 'LG-55UP7000', 'name': '55" UHD 4K UP7000 Smart TV', 'category': 'tvs', 'brand': 'lg', 'price': 499},
        ]
        
        for product in additional_products:
            self.product_catalog[product['sku']] = {
                'sku': product['sku'],
                'product_name': product['name'],
                'product_category': product['category'],
                'product_family': product['category'],
                'brand': product['brand']
            }
            
            # Add to category products
            if product['category'] not in self.category_products:
                self.category_products[product['category']] = {}
            if product['brand'] not in self.category_products[product['category']]:
                self.category_products[product['category']][product['brand']] = []
            self.category_products[product['category']][product['brand']].append(self.product_catalog[product['sku']])
        
        return pd.DataFrame(sample_data)
    
    def detect_brand(self, product_name, sku):
        """Detect brand from product name or SKU"""
        product_name_lower = product_name.lower()
        sku_lower = sku.lower()
        
        brand_indicators = {
            'samsung': ['galaxy', 'samsung', 'sm-', 'qn', 'un-', 'rf23'],
            'apple': ['iphone', 'ipad', 'airpods', 'apple', 'mac', 'watch'],
            'google': ['pixel', 'nest', 'chromecast', 'google'],
            'amazon': ['echo', 'fire', 'kindle', 'alexa'],
            'sony': ['sony', 'playstation', 'ps5', 'wh-', 'xperia'],
            'microsoft': ['surface', 'xbox', 'microsoft'],
            'lg': ['lg-', 'oled', 'lg '],
            'whirlpool': ['whirlpool', 'wrf', 'wmc'],
            'ge': ['ge-', 'general electric'],
            'huawei': ['huawei', 'honor'],
            'oneplus': ['oneplus', 'nord']
        }
        
        for brand, indicators in brand_indicators.items():
            for indicator in indicators:
                if indicator in product_name_lower or indicator in sku_lower:
                    return brand
        
        return 'unknown'
    
    def normalize_category(self, raw_category):
        """Normalize category names for consistency"""
        category_mapping = {
            'phones': ['phones', 'smartphone', 'mobile', 'cell phone'],
            'tablets': ['tablets', 'tablet', 'ipad'],
            'watches': ['watches', 'smartwatch', 'wearables', 'fitness'],
            'audio': ['audio', 'headphones', 'earbuds', 'speakers', 'buds'],
            'tvs': ['tvs', 'television', 'smart tv', 'display'],
            'laptops': ['laptops', 'notebook', 'computer', 'pc'],
            'gaming': ['gaming', 'console', 'xbox', 'playstation'],
            'smart_home': ['smart home', 'home automation', 'iot'],
            'appliances': ['appliances', 'refrigerator', 'microwave', 'washer', 'dryer', 'dishwasher'],
            'tv_streaming': ['tv streaming', 'streaming', 'apple tv'],
            'accessories': ['accessories', 'cases', 'chargers', 'cables']
        }
        
        raw_category_lower = raw_category.lower()
        for normalized, variants in category_mapping.items():
            if any(variant in raw_category_lower for variant in variants):
                return normalized
        
        return raw_category_lower
    
    def get_family_brand_preference(self, family_id):
        """Determine family's primary brand preference"""
        if self.reference_data is None:
            return 'mixed'
        
        family_data = self.reference_data[self.reference_data['family_id'] == family_id]
        
        if len(family_data) == 0:
            return 'mixed'
        
        # Calculate brand spending
        brand_spending = family_data.groupby('brand')['price'].sum().sort_values(ascending=False)
        total_spending = brand_spending.sum()
        
        if len(brand_spending) == 0 or total_spending == 0:
            return 'mixed'
        
        primary_brand = brand_spending.index[0]
        primary_brand_ratio = brand_spending.iloc[0] / total_spending
        
        # If 60%+ of spending is on one brand, consider it primary preference
        return primary_brand if primary_brand_ratio >= 0.6 else 'mixed'
    
    def build_association_rules(self, df, min_support=0.05, min_confidence=0.1):
        """Build association rules for ecosystem completion"""
        print("🔗 Building association rules for product ecosystem analysis...")
        
        # Create transaction data (family -> list of categories)
        family_transactions = []
        for family_id in df['family_id'].unique():
            family_data = df[df['family_id'] == family_id]
            categories = list(family_data['product_category'].unique())
            
            if len(categories) >= 2:  # Only families with multiple product categories
                family_transactions.append(categories)
        
        if len(family_transactions) < 2:
            print("⚠️ Not enough multi-category families for association rules")
            self.association_rules_model = pd.DataFrame()
            return
        
        # Apply Transaction Encoder
        te = TransactionEncoder()
        te_array = te.fit(family_transactions).transform(family_transactions)
        transaction_df = pd.DataFrame(te_array, columns=te.columns_)
        
        # Adjust min_support based on data size
        min_support = max(min_support, 2.0 / len(family_transactions))
        
        try:
            # Generate frequent itemsets
            frequent_itemsets = fpgrowth(transaction_df, min_support=min_support, use_colnames=True)
            
            if len(frequent_itemsets) > 0:
                # Generate association rules
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                self.association_rules_model = rules
                print(f"✅ Generated {len(rules)} association rules")
            else:
                self.association_rules_model = pd.DataFrame()
                print("⚠️ No frequent itemsets found")
                
        except Exception as e:
            print(f"⚠️ Association rules generation failed: {e}")
            self.association_rules_model = pd.DataFrame()
    
    def build_price_predictor(self, df):
        """Build price band prediction model using XGBoost"""
        print("💰 Building price band prediction model...")
        
        # Calculate family spending patterns
        family_stats = df.groupby(['family_id', 'product_category']).agg({
            'price': ['mean', 'max', 'min', 'std', 'count'],
            'days_since_purchase': 'mean'
        }).reset_index()
        
        family_stats.columns = [
            'family_id', 'product_category', 'avg_price', 'max_price', 'min_price',
            'price_std', 'purchase_count', 'avg_days_owned'
        ]
        family_stats['price_std'] = family_stats['price_std'].fillna(0)
        
        # Calculate overall family spending power
        family_totals = df.groupby('family_id').agg({
            'price': ['sum', 'mean', 'count']
        }).reset_index()
        family_totals.columns = ['family_id', 'total_spending', 'avg_item_price', 'total_items']
        
        # Merge data
        model_data = family_stats.merge(family_totals, on='family_id')
        
        # Encode categorical variables
        le_category = LabelEncoder()
        model_data['category_encoded'] = le_category.fit_transform(model_data['product_category'])
        self.label_encoders['product_category'] = le_category
        
        # Prepare features
        feature_cols = [
            'category_encoded', 'max_price', 'min_price', 'price_std', 'purchase_count',
            'avg_days_owned', 'total_spending', 'avg_item_price', 'total_items'
        ]
        
        X = model_data[feature_cols]
        y = model_data['avg_price']
        
        if len(X) >= 5:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Use XGBoost for price prediction
            self.price_predictor = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            self.price_predictor.fit(X_train, y_train)
            
            train_score = self.price_predictor.score(X_train, y_train)
            test_score = self.price_predictor.score(X_test, y_test)
            print(f"✅ Price predictor - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
        else:
            print("⚠️ Using simple linear regression due to limited data")
            from sklearn.linear_model import LinearRegression
            self.price_predictor = LinearRegression()
            self.price_predictor.fit(X, y)
        
        # Build price bands by category
        self.price_bands = {}
        for category in df['product_category'].unique():
            category_data = df[df['product_category'] == category]
            self.price_bands[category] = {
                'min': category_data['price'].quantile(0.1),
                'max': category_data['price'].quantile(0.9),
                'mean': category_data['price'].mean(),
                'std': category_data['price'].std()
            }
        
        print(f"✅ Built price bands for {len(self.price_bands)} categories")
    
    def build_survival_model(self, df):
        """Build survival analysis model for upgrade timing"""
        print("⏰ Building survival model for lifecycle analysis...")
        
        # Category-specific typical lifespans (in days)
        typical_lifespans = {
            'phones': 730,      # 2 years
            'tablets': 1095,    # 3 years
            'laptops': 1460,    # 4 years
            'watches': 1095,    # 3 years
            'audio': 730,       # 2 years
            'tvs': 2555,        # 7 years
            'gaming': 1825,     # 5 years
            'appliances': 3650  # 10 years
        }
        
        self.survival_model = {}
        
        for category in df['product_category'].unique():
            category_data = df[df['product_category'] == category]
            
            if len(category_data) >= 3:
                try:
                    # Try to fit Weibull model
                    durations = category_data['days_since_purchase']
                    events = [1 if days > typical_lifespans.get(category, 1095) * 0.7 else 0 for days in durations]
                    
                    wf = WeibullFitter()
                    wf.fit(durations, events)
                    self.survival_model[category] = wf
                    
                except Exception:
                    # Fallback to heuristic model
                    avg_days = category_data['days_since_purchase'].mean()
                    typical_lifespan = typical_lifespans.get(category, 1095)
                    
                    self.survival_model[category] = {
                        'type': 'heuristic',
                        'avg_lifespan': max(avg_days, typical_lifespan),
                        'upgrade_threshold': max(avg_days * 0.7, typical_lifespan * 0.6)
                    }
            else:
                # Use heuristic for categories with limited data
                typical_lifespan = typical_lifespans.get(category, 1095)
                self.survival_model[category] = {
                    'type': 'heuristic',
                    'avg_lifespan': typical_lifespan,
                    'upgrade_threshold': typical_lifespan * 0.7
                }
        
        print(f"✅ Built survival models for {len(self.survival_model)} categories")
    
    def predict_missing_products(self, family_id):
        """Predict missing products using association rules and brand ecosystem"""
        recommendations = []
        
        if self.reference_data is None:
            return recommendations
        
        family_data = self.reference_data[self.reference_data['family_id'] == family_id]
        if len(family_data) == 0:
            return recommendations
        
        owned_categories = set(family_data['product_category'].unique())
        preferred_brand = self.get_family_brand_preference(family_id)
        
        # Use association rules if available
        if self.association_rules_model is not None and len(self.association_rules_model) > 0:
            for _, rule in self.association_rules_model.iterrows():
                antecedent = set(rule['antecedents'])
                consequent = set(rule['consequents'])
                confidence = rule['confidence']
                
                if (antecedent.issubset(owned_categories) and 
                    not consequent.issubset(owned_categories)):
                    
                    for missing_category in consequent - owned_categories:
                        price_info = self.predict_price_band(family_id, missing_category)
                        product_suggestions = self.get_product_suggestions(missing_category, price_info, preferred_brand)
                        
                        recommendations.append({
                            'category': missing_category,
                            'confidence': confidence,
                            'reason': f"Families with {', '.join(antecedent)} typically also buy {missing_category}",
                            'products': product_suggestions,
                            'price_range': price_info
                        })
        
        # Check brand ecosystem completion
        if preferred_brand in self.brand_ecosystems:
            ecosystem_categories = set(self.brand_ecosystems[preferred_brand])
            missing_from_ecosystem = ecosystem_categories - owned_categories
            
            for missing_category in missing_from_ecosystem:
                # Skip if already recommended
                if any(rec['category'] == missing_category for rec in recommendations):
                    continue
                
                price_info = self.predict_price_band(family_id, missing_category)
                product_suggestions = self.get_product_suggestions(missing_category, price_info, preferred_brand)
                
                if product_suggestions:
                    recommendations.append({
                        'category': missing_category,
                        'confidence': 0.4,
                        'reason': f"Complete your {preferred_brand.title()} ecosystem with {missing_category}",
                        'products': product_suggestions,
                        'price_range': price_info
                    })
        
        # Sort by confidence and return top recommendations
        recommendations = sorted(recommendations, key=lambda x: x['confidence'], reverse=True)
        return recommendations[:5]
    
    def predict_price_band(self, family_id, category):
        """Predict affordable price range for a family and category"""
        if self.reference_data is None:
            return self.get_default_price_range(category)
        
        family_data = self.reference_data[self.reference_data['family_id'] == family_id]
        
        if len(family_data) == 0:
            return self.get_default_price_range(category)
        
        # Calculate family's affordability
        family_avg_price = family_data['price'].mean()
        family_max_price = family_data['price'].max()
        
        # Get category price statistics
        if category in self.price_bands:
            category_stats = self.price_bands[category]
            category_avg = category_stats['mean']
            
            # Calculate affordability ratio
            affordability_ratio = min(family_avg_price / category_avg, 1.5) if category_avg > 0 else 1.0
            affordability_ratio = max(affordability_ratio, 0.5)  # Minimum 50% of category average
            
            suggested_min = max(category_stats['min'] * affordability_ratio, category_stats['min'] * 0.5)
            suggested_max = min(category_stats['max'] * affordability_ratio, family_max_price * 1.2)
            
            # Ensure min < max
            if suggested_min >= suggested_max:
                suggested_min = category_avg * affordability_ratio * 0.8
                suggested_max = category_avg * affordability_ratio * 1.2
            
            return {
                'min': max(0, suggested_min),
                'max': suggested_max,
                'suggested': category_avg * affordability_ratio
            }
        
        return self.get_default_price_range(category)
    
    def get_default_price_range(self, category):
        """Get default price ranges by category"""
        defaults = {
            'phones': {'min': 200, 'max': 1200, 'suggested': 600},
            'tablets': {'min': 150, 'max': 800, 'suggested': 400},
            'watches': {'min': 100, 'max': 500, 'suggested': 300},
            'audio': {'min': 50, 'max': 400, 'suggested': 200},
            'tvs': {'min': 300, 'max': 2500, 'suggested': 800},
            'laptops': {'min': 500, 'max': 2500, 'suggested': 1200},
            'gaming': {'min': 300, 'max': 600, 'suggested': 450},
            'appliances': {'min': 200, 'max': 3000, 'suggested': 1000},
            'tv_streaming': {'min': 50, 'max': 200, 'suggested': 120}
        }
        
        return defaults.get(category, {'min': 100, 'max': 1000, 'suggested': 300})
    
    def get_product_suggestions(self, category, price_info, preferred_brand):
        """Get specific product suggestions within price range"""
        suggestions = []
        
        if category not in self.category_products:
            return suggestions
        
        price_min = price_info['min'] if price_info else 0
        price_max = price_info['max'] if price_info else float('inf')
        
        # Prioritize preferred brand
        brands_to_check = []
        if preferred_brand and preferred_brand in self.category_products[category]:
            brands_to_check.append(preferred_brand)
        
        # Add other brands
        for brand in self.category_products[category].keys():
            if brand != preferred_brand:
                brands_to_check.append(brand)
        
        for brand in brands_to_check[:3]:  # Check top 3 brands
            if brand not in self.category_products[category]:
                continue
            
            brand_products = self.category_products[category][brand]
            
            for product in brand_products:
                # Get estimated price (you could enhance this with real pricing data)
                estimated_price = self.estimate_product_price(product, category)
                
                if price_min <= estimated_price <= price_max:
                    suggestions.append({
                        'sku': product['sku'],
                        'name': product['product_name'],
                        'brand': product['brand'],
                        'price': estimated_price,
                        'is_preferred_brand': brand == preferred_brand
                    })
        
        # Sort by brand preference, then price
        suggestions.sort(key=lambda x: (not x['is_preferred_brand'], x['price']))
        return suggestions[:3]  # Return top 3 suggestions
    
    def estimate_product_price(self, product, category):
        """Estimate product price based on category and brand"""
        # Enhanced price estimation logic
        base_prices = {
            'phones': {'samsung': 650, 'apple': 750, 'google': 550, 'unknown': 400},
            'tablets': {'samsung': 500, 'apple': 550, 'google': 400, 'unknown': 300},
            'watches': {'samsung': 350, 'apple': 400, 'unknown': 250},
            'audio': {'samsung': 150, 'apple': 200, 'sony': 300, 'unknown': 120},
            'tvs': {'samsung': 800, 'lg': 900, 'unknown': 600},
            'laptops': {'samsung': 1100, 'apple': 1300, 'unknown': 900},
            'appliances': {'samsung': 1500, 'whirlpool': 1200, 'ge': 1100, 'unknown': 800},
            'gaming': {'sony': 500, 'microsoft': 450, 'unknown': 400},
            'tv_streaming': {'apple': 150, 'google': 100, 'amazon': 80, 'unknown': 90}
        }
        
        category_prices = base_prices.get(category, {'unknown': 300})
        brand = product.get('brand', 'unknown')
        
        return category_prices.get(brand, category_prices.get('unknown', 300))
    
    def predict_upgrades(self, family_id):
        """Predict which products need upgrades based on age and lifecycle"""
        upgrades = []
        
        if self.reference_data is None:
            return upgrades
        
        family_data = self.reference_data[self.reference_data['family_id'] == family_id]
        preferred_brand = self.get_family_brand_preference(family_id)
        
        for category in family_data['product_category'].unique():
            category_data = family_data[family_data['product_category'] == category].sort_values('purchase_date')
            latest_purchase = category_data.iloc[-1]  # Most recent purchase in this category
            
            days_owned = latest_purchase['days_since_purchase']
            
            # Get upgrade urgency from survival model
            if category in self.survival_model:
                survival_info = self.survival_model[category]
                
                if isinstance(survival_info, dict) and survival_info.get('type') == 'heuristic':
                    threshold_days = survival_info['upgrade_threshold']
                    avg_lifespan = survival_info['avg_lifespan']
                    urgency_score = min(days_owned / threshold_days, 1.0) if threshold_days > 0 else 0
                else:
                    try:
                        survival_prob = survival_info.survival_function_at_times([days_owned]).iloc[0]
                        urgency_score = 1 - survival_prob
                    except:
                        urgency_score = min(days_owned / 730, 1.0)  # Default 2-year cycle
                
                if urgency_score > 0.3:  # Only suggest upgrades above 30% urgency
                    # Get upgrade suggestions (products 10-50% more expensive)
                    current_price = latest_purchase['price']
                    upgrade_price_info = {
                        'min': current_price * 1.1,
                        'max': current_price * 1.5,
                        'suggested': current_price * 1.3
                    }
                    
                    upgrade_products = self.get_product_suggestions(category, upgrade_price_info, preferred_brand)
                    
                    # Filter out current product
                    upgrade_products = [p for p in upgrade_products if p['sku'] != latest_purchase['sku']]
                    
                    if urgency_score > 0.8:
                        priority = "URGENT"
                    elif urgency_score > 0.6:
                        priority = "HIGH"
                    elif urgency_score > 0.4:
                        priority = "MEDIUM"
                    else:
                        priority = "LOW"
                    
                    upgrades.append({
                        'category': category,
                        'current_sku': latest_purchase['sku'],
                        'current_name': latest_purchase['product_name'],
                        'current_brand': latest_purchase['brand'],
                        'current_price': current_price,
                        'days_owned': days_owned,
                        'urgency_score': urgency_score,
                        'priority': priority,
                        'upgrade_products': upgrade_products,
                        'trade_in_value': self.estimate_trade_in_value(current_price, days_owned, category)
                    })
        
        # Sort by urgency
        upgrades.sort(key=lambda x: x['urgency_score'], reverse=True)
        return upgrades
    
    def estimate_trade_in_value(self, original_price, days_owned, category):
        """Estimate trade-in value based on depreciation"""
        # Category-specific annual depreciation rates
        depreciation_rates = {
            'phones': 0.4,      # 40% per year
            'tablets': 0.35,    # 35% per year
            'laptops': 0.3,     # 30% per year
            'watches': 0.45,    # 45% per year
            'audio': 0.5,       # 50% per year
            'tvs': 0.25,        # 25% per year
            'gaming': 0.35,     # 35% per year
            'appliances': 0.15  # 15% per year
        }
        
        annual_depreciation = depreciation_rates.get(category, 0.35)
        years_owned = days_owned / 365
        
        # Apply depreciation with minimum floor
        depreciated_value = original_price * (1 - annual_depreciation) ** years_owned
        min_value = original_price * 0.1  # Minimum 10% of original
        
        return max(depreciated_value, min_value)
    
    def train_pipeline(self, catalog_path, lineitem_path, orderdetail_path):
        """Train the complete ML pipeline"""
        print("🚀 TRAINING AI FAMILY RECOMMENDATION ENGINE")
        print("=" * 60)
        
        # Load and preprocess data
        df = self.load_and_preprocess_data(catalog_path, lineitem_path, orderdetail_path)
        
        # Store reference data FIRST before building models
        self.reference_data = df
        
        # Build all models
        self.build_association_rules(df)
        self.build_price_predictor(df)
        self.build_survival_model(df)
        
        print("✅ TRAINING COMPLETE!")
        print("=" * 60)
        
        return self
    
    def generate_recommendations(self, family_id):
        """Generate complete recommendations for a family"""
        print(f"\n🎯 Generating AI recommendations for family: {family_id}")
        print("-" * 60)
        
        missing_products = self.predict_missing_products(family_id)
        upgrades = self.predict_upgrades(family_id)
        
        return {
            'family_id': family_id,
            'missing_products': missing_products,
            'upgrades': upgrades,
            'preferred_brand': self.get_family_brand_preference(family_id)
        }
    
    def print_family_purchases(self, family_id):
        """Print all products purchased by a family"""
        if self.reference_data is None:
            print("❌ No data available")
            return
        
        family_data = self.reference_data[self.reference_data['family_id'] == family_id]
        
        if len(family_data) == 0:
            print(f"❌ No purchases found for family: {family_id}")
            return
        
        preferred_brand = self.get_family_brand_preference(family_id)
        total_spent = family_data['price'].sum()
        
        print(f"\n🛍️ PURCHASE HISTORY - FAMILY: {family_id}")
        print(f"🏷️ Primary Brand: {preferred_brand.upper()}")
        print(f"💰 Total Spent: ${total_spent:.2f}")
        print("=" * 80)
        
        # Group by category for cleaner display
        for category in family_data['product_category'].unique():
            category_data = family_data[family_data['product_category'] == category]
            
            for _, purchase in category_data.iterrows():
                brand_indicator = "⭐" if purchase['brand'] == preferred_brand else "🔸"
                
                print(f"{brand_indicator} {purchase['product_name']}")
                print(f"   💰 Price: ${purchase['price']:.2f}")
                print(f"   🏷️ Category: {purchase['product_category'].title()} | Brand: {purchase['brand'].title()}")
                print(f"   📦 SKU: {purchase['sku']}")
                print(f"   📅 Owned for: {purchase['days_since_purchase']} days")
                print()
    
    def print_recommendations(self, family_id):
        """Print formatted recommendations"""
        recs = self.generate_recommendations(family_id)
        
        print(f"\n🎯 AI RECOMMENDATIONS FOR FAMILY: {family_id}")
        print("=" * 80)
        
        # Missing Products (Ecosystem Completion)
        if recs['missing_products']:
            print("\n🔍 ECOSYSTEM COMPLETION - Missing Products:")
            print("-" * 50)
            for i, rec in enumerate(recs['missing_products'], 1):
                price_range = rec['price_range']
                price_str = f"${price_range['min']:.0f} - ${price_range['max']:.0f}" if price_range else "TBD"
                
                print(f"{i}. CATEGORY: {rec['category'].upper()}")
                print(f"   🎯 Confidence: {rec['confidence']:.1%}")
                print(f"   💰 Budget Range: {price_str}")
                print(f"   📝 Reason: {rec['reason']}")
                
                if rec['products']:
                    print("   🛍️ RECOMMENDED PRODUCTS:")
                    for j, product in enumerate(rec['products'], 1):
                        brand_star = "⭐" if product['is_preferred_brand'] else "🔸"
                        print(f"      {j}. {brand_star} {product['name']}")
                        print(f"         💵 Est. Price: ${product['price']:.2f}")
                        print(f"         🏷️ Brand: {product['brand'].title()}")
                        print(f"         📦 SKU: {product['sku']}")
                print()
        
        # Upgrades
        if recs['upgrades']:
            print("\n⬆️ UPGRADE RECOMMENDATIONS:")
            print("-" * 50)
            for i, upgrade in enumerate(recs['upgrades'], 1):
                print(f"{i}. CATEGORY: {upgrade['category'].upper()} - {upgrade['priority']} PRIORITY")
                print(f"   📱 CURRENT: {upgrade['current_name']} ({upgrade['current_brand'].title()})")
                print(f"   💰 Current Value: ${upgrade['current_price']:.2f}")
                print(f"   📅 Owned for: {upgrade['days_owned']} days")
                print(f"   🚨 Urgency: {upgrade['urgency_score']:.1%}")
                print(f"   💸 Est. Trade-in Value: ${upgrade['trade_in_value']:.2f}")
                
                if upgrade['upgrade_products']:
                    print("   🆙 UPGRADE OPTIONS:")
                    for j, product in enumerate(upgrade['upgrade_products'], 1):
                        brand_star = "⭐" if product['is_preferred_brand'] else "🔸"
                        net_cost = product['price'] - upgrade['trade_in_value']
                        print(f"      {j}. {brand_star} {product['name']}")
                        print(f"         💵 Price: ${product['price']:.2f}")
                        print(f"         💸 Net Cost (after trade): ${net_cost:.2f}")
                        print(f"         🏷️ Brand: {product['brand'].title()}")
                        print(f"         📦 SKU: {product['sku']}")
                print()
        
        if not recs['missing_products'] and not recs['upgrades']:
            print("✅ This family's ecosystem looks complete and up-to-date!")
        
        print("=" * 80)
    
    def print_complete_report(self, family_id):
        """Print complete family analysis report"""
        print("\n" + "🎯" * 30)
        print("AI FAMILY RECOMMENDATION REPORT")
        print("🎯" * 30)
        
        self.print_family_purchases(family_id)
        self.print_recommendations(family_id)
        
        print("\n" + "🎯" * 30)
        print("REPORT COMPLETE")
        print("🎯" * 30)

# Main execution and testing
def main():
    """Main function to run the recommendation engine"""
    print("🚀 INITIALIZING AI FAMILY RECOMMENDATION ENGINE")
    print("=" * 60)
    
    # Initialize the engine
    engine = AIFamilyRecommendationEngine()
    
    # File paths (replace with your actual paths)
    catalog_path = "data/catalog.csv"
    lineitem_path = "data/lineitem.csv"
    orderdetail_path = "data/orderdetail.csv"
    
    try:
        # Train the pipeline
        engine.train_pipeline(catalog_path, lineitem_path, orderdetail_path)
        
        # Get sample families for analysis
        if engine.reference_data is not None:
            sample_families = engine.reference_data['family_id'].unique()[:3]
            
            for family_id in sample_families:
                engine.print_complete_report(family_id)
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("📝 Using sample data for demonstration...")
        
        # Train with sample data
        engine.train_pipeline("", "", "")
        
        # Generate reports for sample families
        sample_families = ['family_samsung_tech', 'family_apple_lovers', 'family_mixed_premium', 
                          'family_budget_smart', 'family_tech_upgrader']
        
        for family_id in sample_families:
            engine.print_complete_report(family_id)
    
    print("\n🎯 AI RECOMMENDATION ENGINE DEMO COMPLETE!")

if __name__ == "__main__":
    main()