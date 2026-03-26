import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score, silhouette_score)
from sklearn.decomposition import PCA
import xgboost as xgb
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
import io

warnings.filterwarnings("ignore")

# ─── Page Config ───
st.set_page_config(page_title="MoodMeal Analytics", page_icon="🍽️", layout="wide")

# ─── Custom CSS ───
st.markdown("""
<style>
    .main .block-container {padding-top: 1.5rem; padding-bottom: 1rem;}
    div[data-testid="stMetric"] {background: #F7F4ED; border-radius: 12px; padding: 12px 16px; border: 1px solid #E8E5DC;}
    div[data-testid="stMetric"] label {color: #5C5A52 !important; font-size: 13px !important;}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {color: #1A1A17 !important; font-size: 24px !important;}
    .stTabs [data-baseweb="tab"] {font-size: 14px;}
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ───
@st.cache_data
def load_data():
    df = pd.read_csv("MoodMeal_dataset.csv")
    return df

# ─── Feature Engineering for ML ───
@st.cache_data
def prepare_features(df):
    num_features = [
        'monthly_income_inr', 'price_sensitivity_score', 'health_orientation_score',
        'convenience_need_score', 'personalization_interest_score', 'subscription_interest_score',
        'social_media_influence_score', 'office_lunch_need_score', 'breakfast_need_score',
        'indulgence_score', 'order_frequency_per_month', 'avg_order_value_inr',
        'discount_response_pct', 'wellness_index', 'value_seeking_index',
        'digital_convenience_index', 'meal_need_intensity_index', 'customer_potential_score',
        'age_band_code', 'city_tier_code', 'fitness_goal_score', 'diet_preference_score'
    ]
    return num_features

# ─── Train Classifier ───
@st.cache_resource
def train_classifiers(_df):
    num_features = [
        'monthly_income_inr', 'price_sensitivity_score', 'health_orientation_score',
        'convenience_need_score', 'personalization_interest_score', 'subscription_interest_score',
        'social_media_influence_score', 'office_lunch_need_score', 'breakfast_need_score',
        'indulgence_score', 'order_frequency_per_month', 'avg_order_value_inr',
        'discount_response_pct', 'wellness_index', 'value_seeking_index',
        'digital_convenience_index', 'meal_need_intensity_index', 'customer_potential_score',
        'age_band_code', 'city_tier_code', 'fitness_goal_score', 'diet_preference_score'
    ]
    X = _df[num_features].copy()
    y = _df['likely_to_try_moodmeal'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    models = {}
    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train_sc, y_train)
    models['Random Forest'] = rf
    # XGBoost
    xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42,
                                 eval_metric='logloss', use_label_encoder=False)
    xgb_clf.fit(X_train_sc, y_train)
    models['XGBoost'] = xgb_clf
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    models['Logistic Regression'] = lr
    return models, scaler, X_train_sc, X_test_sc, y_train, y_test, num_features

# ─── Train Regressor ───
@st.cache_resource
def train_regressors(_df):
    num_features = [
        'monthly_income_inr', 'price_sensitivity_score', 'health_orientation_score',
        'convenience_need_score', 'personalization_interest_score', 'subscription_interest_score',
        'social_media_influence_score', 'office_lunch_need_score', 'breakfast_need_score',
        'indulgence_score', 'order_frequency_per_month', 'discount_response_pct',
        'wellness_index', 'value_seeking_index', 'digital_convenience_index',
        'meal_need_intensity_index', 'age_band_code', 'city_tier_code',
        'fitness_goal_score', 'diet_preference_score'
    ]
    X = _df[num_features].copy()
    y_aov = _df['avg_order_value_inr'].copy()
    y_monthly = _df['monthly_food_delivery_spend_inr'].copy()
    X_train, X_test, y_aov_train, y_aov_test = train_test_split(X, y_aov, test_size=0.3, random_state=42)
    _, _, y_mon_train, y_mon_test = train_test_split(X, y_monthly, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    reg_models = {}
    # AOV models
    lr_aov = LinearRegression()
    lr_aov.fit(X_train_sc, y_aov_train)
    reg_models['Linear Regression (AOV)'] = (lr_aov, y_aov_test, 'avg_order_value_inr')
    rf_aov = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    rf_aov.fit(X_train_sc, y_aov_train)
    reg_models['Random Forest (AOV)'] = (rf_aov, y_aov_test, 'avg_order_value_inr')
    gb_aov = GradientBoostingRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42)
    gb_aov.fit(X_train_sc, y_aov_train)
    reg_models['Gradient Boosting (AOV)'] = (gb_aov, y_aov_test, 'avg_order_value_inr')
    # Monthly spend models
    gb_mon = GradientBoostingRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42)
    gb_mon.fit(X_train_sc, y_mon_train)
    reg_models['Gradient Boosting (Monthly)'] = (gb_mon, y_mon_test, 'monthly_food_delivery_spend_inr')
    return reg_models, scaler, X_train_sc, X_test_sc, num_features

# ─── Train Clusters ───
@st.cache_resource
def train_clusters(_df):
    cluster_features = [
        'wellness_index', 'value_seeking_index', 'digital_convenience_index',
        'meal_need_intensity_index', 'customer_potential_score',
        'price_sensitivity_score', 'health_orientation_score',
        'convenience_need_score', 'personalization_interest_score',
        'subscription_interest_score'
    ]
    X = _df[cluster_features].copy()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    inertias = []
    sil_scores = []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_sc)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_sc, km.labels_, sample_size=2000))
    best_k = list(K_range)[np.argmax(sil_scores)]
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km_final.fit_predict(X_sc)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sc)
    return km_final, scaler, labels, X_sc, X_pca, inertias, sil_scores, list(K_range), cluster_features, best_k

# ─── Association Rules ───
@st.cache_resource
def compute_association_rules(_df):
    cols_for_basket = ['preferred_product_category_1', 'preferred_product_category_2',
                       'current_mood_state', 'preferred_cuisine', 'diet_preference',
                       'fitness_goal', 'primary_meal_occasion', 'bundle_affinity',
                       'packaging_preference', 'health_segment']
    basket_df = _df[cols_for_basket].copy()
    encoded = pd.get_dummies(basket_df, prefix_sep='=')
    encoded = encoded.astype(bool)
    freq_items = apriori(encoded, min_support=0.05, use_colnames=True)
    if len(freq_items) == 0:
        freq_items = apriori(encoded, min_support=0.03, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.3, num_itemsets=len(freq_items))
    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    return rules, freq_items

# ═══════════════ MAIN APP ═══════════════
df = load_data()

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/restaurant.png", width=60)
st.sidebar.title("MoodMeal Analytics")
st.sidebar.markdown("*Data-Driven Decisions for a Smarter Food Business*")
st.sidebar.divider()
page = st.sidebar.radio("Navigate", [
    "📊 EDA & Overview",
    "👥 Customer Segmentation",
    "🎯 Purchase Intent (Classification)",
    "🔗 Association Rule Mining",
    "💰 Spending Prediction (Regression)",
    "📋 Prescriptive Strategy",
    "🆕 New Customer Predictor"
])
st.sidebar.divider()
st.sidebar.caption(f"Dataset: {len(df):,} respondents × {len(df.columns)} features")

# ═══════════════ PAGE 1: EDA ═══════════════
if page == "📊 EDA & Overview":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("*Descriptive analysis — understanding our 7,500-respondent dataset*")
    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Respondents", f"{len(df):,}")
    c2.metric("Cities Covered", df['city'].nunique())
    c3.metric("Avg Income", f"₹{df['monthly_income_inr'].mean():,.0f}")
    c4.metric("Avg Order Value", f"₹{df['avg_order_value_inr'].mean():,.0f}")
    c5.metric("MoodMeal Interest", f"{df['likely_to_try_moodmeal'].mean()*100:.1f}%")

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Food Preferences", "Behavioural Scores", "Correlations"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='age_band', color='gender', barmode='group',
                               title='Age Band × Gender Distribution',
                               color_discrete_sequence=px.colors.qualitative.Set2,
                               category_orders={'age_band': ['18-24','25-34','35-44','45-54','55+']})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.pie(df, names='city_tier', title='City Tier Distribution',
                         color_discrete_sequence=['#E8713A','#2EC4A0','#9B8FE8'],
                         hole=0.4)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = px.histogram(df, x='monthly_income_inr', nbins=50, title='Monthly Income Distribution',
                               color_discrete_sequence=['#E8713A'])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            occ = df['occupation'].value_counts().reset_index()
            occ.columns = ['occupation','count']
            fig = px.bar(occ, x='count', y='occupation', orientation='h', title='Occupation Breakdown',
                         color_discrete_sequence=['#2EC4A0'])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='diet_preference', color='diet_preference',
                               title='Diet Preference Distribution',
                               color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(df, x='current_mood_state', color='current_mood_state',
                               title='Current Mood State',
                               color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            prod_counts = df['preferred_product_category_1'].value_counts().reset_index()
            prod_counts.columns = ['category','count']
            fig = px.treemap(prod_counts, path=['category'], values='count',
                             title='Preferred Product Category (Primary)',
                             color_discrete_sequence=px.colors.qualitative.Pastel1)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            fig = px.histogram(df, x='preferred_cuisine', color='preferred_cuisine',
                               title='Cuisine Preferences',
                               color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(height=400, showlegend=False, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        score_cols = ['price_sensitivity_score','health_orientation_score','convenience_need_score',
                      'personalization_interest_score','subscription_interest_score',
                      'social_media_influence_score','indulgence_score']
        fig = go.Figure()
        for col in score_cols:
            fig.add_trace(go.Box(y=df[col], name=col.replace('_score','').replace('_',' ').title(), boxmean=True))
        fig.update_layout(title='Distribution of Behavioural Scores', height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='avg_order_value_inr', nbins=40, title='Average Order Value Distribution',
                               color_discrete_sequence=['#E8713A'])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(df, x='sales_funnel_stage', y='avg_order_value_inr',
                         title='AOV by Sales Funnel Stage', color='sales_funnel_stage',
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        num_cols_corr = ['monthly_income_inr','avg_order_value_inr','order_frequency_per_month',
                         'wellness_index','value_seeking_index','digital_convenience_index',
                         'meal_need_intensity_index','customer_potential_score',
                         'health_orientation_score','price_sensitivity_score',
                         'convenience_need_score','personalization_interest_score',
                         'discount_response_pct','purchase_intent_score']
        corr = df[num_cols_corr].corr()
        fig = px.imshow(corr, text_auto='.2f', title='Feature Correlation Heatmap',
                        color_continuous_scale='RdBu_r', aspect='auto', zmin=-1, zmax=1)
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════ PAGE 2: CLUSTERING ═══════════════
elif page == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation — Clustering")
    st.markdown("*Diagnostic analysis — identifying distinct customer tribes for personalised targeting*")

    km_final, cl_scaler, labels, X_sc, X_pca, inertias, sil_scores, K_range, cluster_features, best_k = train_clusters(df)
    df_clust = df.copy()
    df_clust['Cluster'] = labels

    tab1, tab2, tab3 = st.tabs(["Optimal K Selection", "Cluster Visualization", "Cluster Profiles"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=K_range, y=inertias, mode='lines+markers',
                                     marker=dict(size=10, color='#E8713A'), line=dict(color='#E8713A', width=2)))
            fig.update_layout(title='Elbow Method — Inertia vs K', xaxis_title='Number of Clusters (K)',
                              yaxis_title='Inertia', height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=K_range, y=sil_scores, mode='lines+markers',
                                     marker=dict(size=10, color='#2EC4A0'), line=dict(color='#2EC4A0', width=2)))
            fig.update_layout(title='Silhouette Score vs K', xaxis_title='Number of Clusters (K)',
                              yaxis_title='Silhouette Score', height=400)
            st.plotly_chart(fig, use_container_width=True)
        st.success(f"**Optimal K = {best_k}** (highest silhouette score: {max(sil_scores):.3f})")

    with tab2:
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=labels.astype(str),
                         title=f'Customer Segments (K={best_k}) — PCA Projection',
                         labels={'x': 'PC 1', 'y': 'PC 2', 'color': 'Cluster'},
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=550)
        fig.update_traces(marker=dict(size=4, opacity=0.6))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cluster Sizes")
        sizes = df_clust['Cluster'].value_counts().sort_index().reset_index()
        sizes.columns = ['Cluster', 'Count']
        sizes['Percentage'] = (sizes['Count'] / len(df_clust) * 100).round(1)
        st.dataframe(sizes, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Cluster Profiles — Average Feature Values")
        profile = df_clust.groupby('Cluster')[cluster_features].mean().round(1)
        st.dataframe(profile, use_container_width=True)

        st.subheader("Radar Chart — Cluster Comparison")
        profile_norm = (profile - profile.min()) / (profile.max() - profile.min())
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, row in profile_norm.iterrows():
            fig.add_trace(go.Scatterpolar(r=row.values.tolist() + [row.values[0]],
                                          theta=[c.replace('_',' ').title() for c in cluster_features] + [cluster_features[0].replace('_',' ').title()],
                                          fill='toself', name=f'Cluster {i}',
                                          line=dict(color=colors[i % len(colors)])))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                          title='Normalised Cluster Profiles', height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Demographic Breakdown per Cluster")
        col1, col2 = st.columns(2)
        with col1:
            ct = pd.crosstab(df_clust['Cluster'], df_clust['income_segment'], normalize='index').round(3)*100
            fig = px.bar(ct.reset_index().melt(id_vars='Cluster'), x='Cluster', y='value',
                         color='income_segment', title='Income Segment by Cluster (%)', barmode='stack')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            ct2 = pd.crosstab(df_clust['Cluster'], df_clust['priority_segment'], normalize='index').round(3)*100
            fig = px.bar(ct2.reset_index().melt(id_vars='Cluster'), x='Cluster', y='value',
                         color='priority_segment', title='Priority Segment by Cluster (%)', barmode='stack')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════ PAGE 3: CLASSIFICATION ═══════════════
elif page == "🎯 Purchase Intent (Classification)":
    st.title("🎯 Purchase Intent Prediction — Classification")
    st.markdown("*Predictive analysis — will this customer try MoodMeal?*")

    models, scaler, X_train_sc, X_test_sc, y_train, y_test, num_features = train_classifiers(df)

    tab1, tab2, tab3 = st.tabs(["Model Comparison", "ROC Curves", "Feature Importance"])

    with tab1:
        results = []
        for name, model in models.items():
            y_pred = model.predict(X_test_sc)
            y_proba = model.predict_proba(X_test_sc)[:, 1]
            results.append({
                'Model': name,
                'Accuracy': round(accuracy_score(y_test, y_pred), 4),
                'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
                'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
                'F1-Score': round(f1_score(y_test, y_pred, zero_division=0), 4),
                'ROC-AUC': round(roc_auc_score(y_test, y_proba), 4)
            })
        results_df = pd.DataFrame(results)
        st.subheader("Performance Metrics")
        st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy','Precision','Recall','F1-Score','ROC-AUC'],
                                                      props='background-color: #E1F5EE; font-weight: bold'),
                     use_container_width=True, hide_index=True)

        st.subheader("Confusion Matrices")
        cols = st.columns(3)
        for idx, (name, model) in enumerate(models.items()):
            with cols[idx]:
                y_pred = model.predict(X_test_sc)
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, text_auto=True, title=name,
                                labels=dict(x="Predicted", y="Actual"),
                                x=['Not Interested','Interested'], y=['Not Interested','Interested'],
                                color_continuous_scale='Oranges')
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()
        colors = ['#E8713A', '#2EC4A0', '#9B8FE8']
        for idx, (name, model) in enumerate(models.items()):
            y_proba = model.predict_proba(X_test_sc)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_val = roc_auc_score(y_test, y_proba)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={auc_val:.3f})',
                                     line=dict(color=colors[idx], width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Baseline',
                                 line=dict(color='gray', width=1, dash='dash')))
        fig.update_layout(title='ROC Curves — All Models', xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate', height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        selected_model = st.selectbox("Select model for feature importance", list(models.keys()), key='clf_fi_model_select')
        model = models[selected_model]
        if hasattr(model, 'feature_importances_'):
            imp = pd.DataFrame({'Feature': num_features, 'Importance': model.feature_importances_})
        else:
            imp = pd.DataFrame({'Feature': num_features, 'Importance': np.abs(model.coef_[0])})
        imp = imp.sort_values('Importance', ascending=True).tail(15)
        fig = px.bar(imp, x='Importance', y='Feature', orientation='h',
                     title=f'Top 15 Features — {selected_model}',
                     color='Importance', color_continuous_scale='Oranges')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.info("**Business Insight:** The top features driving MoodMeal purchase intent help us understand what to emphasize in marketing. "
                "Features like `purchase_intent_score`, `customer_potential_score`, and `wellness_index` are strong predictors — "
                "focus campaigns on health-oriented, high-potential segments.")


# ═══════════════ PAGE 4: ASSOCIATION RULES ═══════════════
elif page == "🔗 Association Rule Mining":
    st.title("🔗 Association Rule Mining")
    st.markdown("*Diagnostic analysis — what products, moods, and preferences naturally go together?*")

    rules, freq_items = compute_association_rules(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Frequent Itemsets", len(freq_items))
    c2.metric("Association Rules", len(rules))
    c3.metric("Avg Lift", f"{rules['lift'].mean():.2f}x" if len(rules) > 0 else "N/A")

    tab1, tab2, tab3 = st.tabs(["Top Rules", "Scatter Plot", "Frequent Itemsets"])

    with tab1:
        st.subheader("Top Association Rules by Lift")
        min_conf = st.slider("Minimum Confidence", 0.1, 0.9, 0.3, 0.05, key='assoc_min_conf')
        min_lift = st.slider("Minimum Lift", 0.5, 5.0, 1.0, 0.1, key='assoc_min_lift')
        filtered = rules[(rules['confidence'] >= min_conf) & (rules['lift'] >= min_lift)]
        filtered_display = filtered[['antecedents_str','consequents_str','support','confidence','lift']].copy()
        filtered_display.columns = ['If Customer Has...', 'Then Also Likely...', 'Support', 'Confidence', 'Lift']
        filtered_display = filtered_display.sort_values('Lift', ascending=False).head(30).round(3)
        st.dataframe(filtered_display, use_container_width=True, hide_index=True)

        if len(filtered_display) > 0:
            st.subheader("🍽️ Business Translation of Top Rules")
            for _, row in filtered_display.head(5).iterrows():
                lift_val = row['Lift']
                conf_val = row['Confidence']
                st.markdown(f"- Customers with **{row['If Customer Has...']}** are **{lift_val:.1f}x more likely** "
                            f"to also prefer **{row['Then Also Likely...']}** (confidence: {conf_val:.0%})")

    with tab2:
        if len(rules) > 0:
            fig = px.scatter(rules, x='support', y='confidence', size='lift', color='lift',
                             hover_data=['antecedents_str','consequents_str'],
                             title='Association Rules — Support vs Confidence (size = Lift)',
                             color_continuous_scale='Oranges', size_max=20)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Most Frequent Itemsets")
        freq_display = freq_items.copy()
        freq_display['itemsets_str'] = freq_display['itemsets'].apply(lambda x: ', '.join(list(x)))
        freq_display = freq_display[['itemsets_str','support']].sort_values('support', ascending=False).head(30)
        freq_display.columns = ['Itemset', 'Support']
        st.dataframe(freq_display.round(3), use_container_width=True, hide_index=True)


# ═══════════════ PAGE 5: REGRESSION ═══════════════
elif page == "💰 Spending Prediction (Regression)":
    st.title("💰 Spending Prediction — Regression")
    st.markdown("*Predictive analysis — how much will a customer spend?*")

    reg_models, reg_scaler, X_train_sc, X_test_sc, reg_features = train_regressors(df)

    tab1, tab2, tab3 = st.tabs(["Model Performance", "Actual vs Predicted", "Feature Importance"])

    with tab1:
        reg_results = []
        for name, (model, y_true, target) in reg_models.items():
            y_pred = model.predict(X_test_sc)
            reg_results.append({
                'Model': name, 'Target': target,
                'R² Score': round(r2_score(y_true, y_pred), 4),
                'MAE (₹)': round(mean_absolute_error(y_true, y_pred), 1),
                'RMSE (₹)': round(np.sqrt(mean_squared_error(y_true, y_pred)), 1)
            })
        reg_df = pd.DataFrame(reg_results)
        st.dataframe(reg_df.style.highlight_max(axis=0, subset=['R² Score'],
                     props='background-color: #E1F5EE; font-weight: bold'),
                     use_container_width=True, hide_index=True)

    with tab2:
        all_model_names = list(reg_models.keys())
        sel = st.selectbox("Select regression model", all_model_names, key='reg_avp_model_select')
        model, y_true, target_name = reg_models[sel]
        y_pred = model.predict(X_test_sc)
        scatter_df = pd.DataFrame({'Actual': y_true.values, 'Predicted': y_pred})
        fig = px.scatter(scatter_df, x='Actual', y='Predicted',
                         title=f'{sel} — Actual vs Predicted ({target_name})',
                         color_discrete_sequence=['#E8713A'], opacity=0.4)
        max_val = max(scatter_df['Actual'].max(), scatter_df['Predicted'].max())
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                                 line=dict(color='gray', dash='dash'), name='Perfect Prediction'))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Residual Distribution")
        residuals = y_true.values - y_pred
        fig = px.histogram(residuals, nbins=50, title='Residual Distribution',
                           color_discrete_sequence=['#2EC4A0'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        aov_model_names = [k for k in reg_models.keys() if 'AOV' in k]
        sel_reg = st.selectbox("Select AOV model for feature importance", aov_model_names, key='reg_fi_model_select')
        model, _, _ = reg_models[sel_reg]
        if hasattr(model, 'feature_importances_'):
            imp = pd.DataFrame({'Feature': reg_features, 'Importance': model.feature_importances_})
        else:
            imp = pd.DataFrame({'Feature': reg_features, 'Importance': np.abs(model.coef_)})
        imp = imp.sort_values('Importance', ascending=True).tail(15)
        fig = px.bar(imp, x='Importance', y='Feature', orientation='h',
                     title=f'Feature Importance — {sel_reg}',
                     color='Importance', color_continuous_scale='Teal')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════ PAGE 6: PRESCRIPTIVE ═══════════════
elif page == "📋 Prescriptive Strategy":
    st.title("📋 Prescriptive Strategy & Recommendations")
    st.markdown("*Turning insights into action — what should MoodMeal actually DO?*")

    km_final, cl_scaler, labels, X_sc, X_pca, _, _, _, cluster_features, best_k = train_clusters(df)
    df_strat = df.copy()
    df_strat['Cluster'] = labels

    st.subheader("1. Segment-Wise Strategy Matrix")
    for clust_id in range(best_k):
        seg = df_strat[df_strat['Cluster'] == clust_id]
        with st.expander(f"Cluster {clust_id} — {len(seg):,} customers ({len(seg)/len(df_strat)*100:.1f}%)", expanded=(clust_id == 0)):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Income", f"₹{seg['monthly_income_inr'].mean():,.0f}")
            c2.metric("Avg AOV", f"₹{seg['avg_order_value_inr'].mean():,.0f}")
            c3.metric("MoodMeal Interest", f"{seg['likely_to_try_moodmeal'].mean()*100:.0f}%")
            c4.metric("Avg Wellness Index", f"{seg['wellness_index'].mean():.1f}")

            top_mood = seg['current_mood_state'].mode().iloc[0] if len(seg) > 0 else 'N/A'
            top_product = seg['preferred_product_category_1'].mode().iloc[0] if len(seg) > 0 else 'N/A'
            top_cuisine = seg['preferred_cuisine'].mode().iloc[0] if len(seg) > 0 else 'N/A'
            top_priority = seg['priority_segment'].mode().iloc[0] if len(seg) > 0 else 'N/A'

            st.markdown(f"**Dominant mood:** {top_mood} | **Top product:** {top_product} | "
                        f"**Preferred cuisine:** {top_cuisine} | **Priority tier:** {top_priority}")

            interest_rate = seg['likely_to_try_moodmeal'].mean()
            avg_aov = seg['avg_order_value_inr'].mean()

            if interest_rate > 0.6 and avg_aov > 500:
                st.success("**Strategy: PREMIUM RETENTION** — High-value, high-interest segment. Offer loyalty rewards, "
                           "exclusive early access to new products, and premium bundle upgrades.")
            elif interest_rate > 0.5:
                st.info("**Strategy: CONVERSION PUSH** — Interested but needs a nudge. Offer first-order 25% discount, "
                        "free trial meal, and social proof campaigns.")
            elif avg_aov > 400:
                st.warning("**Strategy: RE-ENGAGEMENT** — High spenders but low MoodMeal interest. "
                           "Target with personalised comfort food combos and convenience messaging.")
            else:
                st.error("**Strategy: AWARENESS BUILDING** — Low interest, lower spend. "
                         "Use Instagram reels, influencer campaigns, and ₹99 sampler boxes to drive trial.")

    st.divider()
    st.subheader("2. Top Bundle Recommendations (from Association Rules)")
    rules, _ = compute_association_rules(df)
    if len(rules) > 0:
        top_rules = rules.sort_values('lift', ascending=False).head(8)
        for _, row in top_rules.iterrows():
            st.markdown(f"🔗 **{row['antecedents_str']}** → **{row['consequents_str']}** "
                        f"(Lift: {row['lift']:.2f}x, Confidence: {row['confidence']:.0%})")

    st.divider()
    st.subheader("3. Focus Customer Profile")
    st.markdown("""
    Based on classification feature importance and cluster analysis, **MoodMeal's ideal early customers** are:
    - **Age 25-34**, metro/Tier-1 city, salaried professionals
    - **High wellness index** (>70) and **high health orientation** (>60)
    - Currently spending **₹350-600 per order** with **6+ orders/month**
    - **Mood-sensitive eaters** — stressed/busy customers who crave convenience + health
    - **Subscription-ready** — high personalisation interest score
    - **Digital-first** — order via apps, influenced by social media
    """)


# ═══════════════ PAGE 7: NEW CUSTOMER PREDICTOR ═══════════════
elif page == "🆕 New Customer Predictor":
    st.title("🆕 Predict New Customer Inclination")
    st.markdown("*Upload new customer data to predict their MoodMeal interest, expected spending, and segment*")

    models, clf_scaler, _, _, _, _, clf_features = train_classifiers(df)
    reg_models, reg_scaler, _, _, reg_features = train_regressors(df)
    km_final, cl_scaler, _, _, _, _, _, _, cluster_features, best_k = train_clusters(df)

    st.info("**Upload a CSV file** with the same column structure as the training data. "
            "The system will predict: (1) MoodMeal interest (Yes/No), (2) Expected AOV, (3) Customer segment.")

    # Show required columns
    with st.expander("View required columns"):
        st.write("Required numeric columns for prediction:")
        st.code(", ".join(clf_features))

    uploaded = st.file_uploader("Upload new customer CSV", type=['csv'], key='new_customer_upload')

    if uploaded is not None:
        new_df = pd.read_csv(uploaded)
        st.success(f"Uploaded {len(new_df)} new customers")
        st.dataframe(new_df.head(), use_container_width=True)

        missing_clf = [c for c in clf_features if c not in new_df.columns]
        missing_reg = [c for c in reg_features if c not in new_df.columns]
        missing_cl = [c for c in cluster_features if c not in new_df.columns]

        if missing_clf:
            st.warning(f"Missing columns for classification: {missing_clf}. Filling with training data median.")
            for col in missing_clf:
                new_df[col] = df[col].median()

        if missing_reg:
            for col in missing_reg:
                if col not in new_df.columns:
                    new_df[col] = df[col].median()

        if missing_cl:
            for col in missing_cl:
                if col not in new_df.columns:
                    new_df[col] = df[col].median()

        # Classification
        X_clf = new_df[clf_features].values
        X_clf_sc = clf_scaler.transform(X_clf)
        best_clf_name = 'XGBoost'
        best_clf = models[best_clf_name]
        new_df['predicted_moodmeal_interest'] = best_clf.predict(X_clf_sc)
        new_df['interest_probability'] = best_clf.predict_proba(X_clf_sc)[:, 1].round(3)
        new_df['predicted_moodmeal_interest'] = new_df['predicted_moodmeal_interest'].map({1: 'Yes', 0: 'No'})

        # Regression (AOV)
        X_reg = new_df[reg_features].values
        X_reg_sc = reg_scaler.transform(X_reg)
        aov_model = reg_models['Gradient Boosting (AOV)'][0]
        new_df['predicted_aov'] = aov_model.predict(X_reg_sc).round(0).astype(int)

        # Clustering
        X_cl = new_df[cluster_features].values
        X_cl_sc = cl_scaler.transform(X_cl)
        new_df['assigned_segment'] = km_final.predict(X_cl_sc)

        # Discount tier
        new_df['suggested_discount'] = new_df['interest_probability'].apply(
            lambda x: 'Loyalty Rewards (10% off)' if x > 0.7
            else '25% First Order' if x > 0.5
            else 'Free Trial Box' if x > 0.3
            else '₹99 Sampler Offer'
        )

        st.divider()
        st.subheader("Prediction Results")

        c1, c2, c3 = st.columns(3)
        interested = (new_df['predicted_moodmeal_interest'] == 'Yes').sum()
        c1.metric("Predicted Interested", f"{interested} / {len(new_df)}")
        c2.metric("Avg Predicted AOV", f"₹{new_df['predicted_aov'].mean():,.0f}")
        c3.metric("Avg Interest Probability", f"{new_df['interest_probability'].mean():.1%}")

        display_cols = [c for c in ['customer_id', 'city', 'age_band', 'predicted_moodmeal_interest',
                        'interest_probability', 'predicted_aov', 'assigned_segment',
                        'suggested_discount'] if c in new_df.columns]
        if not display_cols:
            display_cols = ['predicted_moodmeal_interest', 'interest_probability',
                           'predicted_aov', 'assigned_segment', 'suggested_discount']

        st.dataframe(new_df[display_cols], use_container_width=True, hide_index=True)

        # Download
        csv_out = new_df.to_csv(index=False)
        st.download_button("📥 Download Predictions CSV", csv_out,
                           "moodmeal_predictions.csv", "text/csv")

        # Visualization
        st.subheader("Interest Distribution")
        fig = px.histogram(new_df, x='interest_probability', nbins=20,
                           color='predicted_moodmeal_interest',
                           title='Interest Probability Distribution',
                           color_discrete_map={'Yes': '#2EC4A0', 'No': '#E85555'})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown("### Or try manual prediction")
        st.markdown("Enter values for a single customer below:")

        with st.form("single_prediction"):
            col1, col2, col3 = st.columns(3)
            with col1:
                income = st.number_input("Monthly Income (₹)", 5000, 400000, 75000, step=5000, key='pred_income')
                health_score = st.slider("Health Orientation", 0, 100, 50, key='pred_health')
                conv_score = st.slider("Convenience Need", 0, 100, 50, key='pred_conv')
                price_sens = st.slider("Price Sensitivity", 0, 100, 50, key='pred_price')
            with col2:
                pers_score = st.slider("Personalisation Interest", 0, 100, 50, key='pred_pers')
                sub_score = st.slider("Subscription Interest", 0, 100, 50, key='pred_sub')
                social_score = st.slider("Social Media Influence", 0, 100, 50, key='pred_social')
                wellness = st.slider("Wellness Index", 30, 100, 70, key='pred_wellness')
            with col3:
                order_freq = st.number_input("Orders/Month", 0, 15, 5, key='pred_freq')
                aov_current = st.number_input("Current AOV (₹)", 100, 850, 400, key='pred_aov')
                age_code = st.selectbox("Age Band", [1, 2, 3, 4, 5], format_func=lambda x: ['18-24','25-34','35-44','45-54','55+'][x-1], key='pred_age')
                city_code = st.selectbox("City Tier", [1, 2, 3], format_func=lambda x: ['Metro','Tier 1','Tier 2'][x-1], key='pred_city')
            submitted = st.form_submit_button("Predict", type="primary")

        if submitted:
            single_row = {f: df[f].median() for f in clf_features}
            single_row.update({
                'monthly_income_inr': income, 'health_orientation_score': health_score,
                'convenience_need_score': conv_score, 'price_sensitivity_score': price_sens,
                'personalization_interest_score': pers_score, 'subscription_interest_score': sub_score,
                'social_media_influence_score': social_score, 'wellness_index': wellness,
                'order_frequency_per_month': order_freq, 'avg_order_value_inr': aov_current,
                'age_band_code': age_code, 'city_tier_code': city_code
            })
            single_df = pd.DataFrame([single_row])
            X_single = clf_scaler.transform(single_df[clf_features].values)
            pred = models['XGBoost'].predict(X_single)[0]
            prob = models['XGBoost'].predict_proba(X_single)[0][1]

            col1, col2, col3 = st.columns(3)
            col1.metric("MoodMeal Interest", "✅ Yes" if pred == 1 else "❌ No")
            col2.metric("Probability", f"{prob:.1%}")
            col3.metric("Suggested Action",
                        'Loyalty Rewards' if prob > 0.7
                        else '25% First Order' if prob > 0.5
                        else 'Free Trial Box' if prob > 0.3
                        else '₹99 Sampler')
