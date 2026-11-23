# MandiMitra: AI-Powered Agricultural Market Intelligence & Farmer Assistance Application

**MandiMitra (मंडीमित्र / ਮੰਡੀਮਿੱਤਰ)** is a production-oriented, mobile-first agricultural market decision-support application built for Indian farmers, FPOs, agricultural traders, and researchers.

Based on the **SIH26132** problem statement and unsupervised machine learning research, MandiMitra bridges the gap between complex econometric data and everyday farmer decisions:
1. **Where should I sell my crop?** (Transport-adjusted net price ranking)
2. **When should I sell my crop?** (Sell now vs wait assistant with storage decay modeling)
3. **Which mandi offers the highest real profit?** (Accounting for distance, freight, and mandi fees)
4. **Are prices increasing or decreasing?** (Mathematical momentum and volatility classification)
5. **How does market price compare to MSP?** (Live comparison with official Government MSP benchmarks)

---

## Architecture Overview

MandiMitra is built with **Two Connected Experiences** operating on a unified **One-Commodity-at-a-Time Analysis Architecture**:

```
                       ┌──────────────────────────────────────────────┐
                       │           Active Commodity Selector          │
                       │ (Wheat, Rice, Mustard, Maize, Potato, etc.)  │
                       └──────────────────────┬───────────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      │    Commodity Data Store & Isolation Layer     │
                      │    data/commodities/<Commodity>/              │
                      │    - raw_data.csv.gz     - market_snapshot    │
                      │    - results_summary     - figures/           │
                      └───────────────┬───────────────────────────────┘
                                      │
           ┌──────────────────────────┴──────────────────────────┐
           │                                                     │
           ▼                                                     ▼
┌─────────────────────────────────────┐   ┌─────────────────────────────────────┐
│    EXPERIENCE 1: FARMER APP         │   │   EXPERIENCE 2: RESEARCH & INTEL    │
│  (Mobile-First Practical Decision)  │   │   (All 4 Objectives & 14 Tabs)      │
├─────────────────────────────────────┤   ├─────────────────────────────────────┤
│ 1. Dashboard & Local Weather        │   │ Objective 1: Preprocessing          │
│ 2. MandiMitra AI (Where & When)     │   │ - Data Profiling, Cleaning Log      │
│ 3. Market Trends (Up/Down Analysis) │   │ - Features, Correlation Pruning     │
│ 4. Profit & Cultivation Calculator  │   │ Objective 2: Segmentation           │
│ 5. Find Mandis & Comparison Map     │   │ - Choosing K, PCA 2D, Ward Tree     │
│ 6. Price Alerts & Watchlist         │   │ Objective 3: Validation             │
│ 7. Farmer Services & Govt MSP       │   │ - Cluster Profiles, Drivers, ARI    │
│ 8. Grounded AI Chat Assistant       │   │ Objective 4: Decision Support       │
│                                     │   │ - Academic Advisor, Download Zip    │
└─────────────────────────────────────┘   └─────────────────────────────────────┘
```

---

## Key Features

### 1. MandiMitra AI: Best Mandi Recommendation
- **Formula:**
  $$\text{Net Price per Quintal} = \text{Listed Market Price} \times \left(1 - \frac{\text{Mandi Fee \%}}{100}\right) - (\text{Transport Rate} \times \text{Road km})$$
  $$\text{Total Realization} = \text{Net Price per Quintal} \times \text{Quantity}$$
- Ranks mandis by **money in the farmer's pocket**, proving why the highest listed nominal price is often inferior once hauling expenses are deducted.
- Highlights quality/grade anomaly warnings if a market lists prices $>1.25\times$ above the regional median.

### 2. Sell Now or Wait Assistant
- Mathematical analysis of price trajectory over 6 weekly reports.
- Computes damped linear slope:
  $$f = P_{\text{current}} + (\text{Slope} \times 2 \times 0.60)$$
- Factors in weekly storage loss (default 0.4%/week) and freight:
  $$\text{Net Gain} = \text{Future Net} - \text{Current Net}$$
- Actionable signals:
  - **Hold 1 to 2 Weeks**: Rising momentum outpaces storage loss.
  - **Sell Now**: Market is softening or flat; holding adds storage loss risk.
  - **Sell or Wait: Close Call**: Price movement is within normal seasonal variance.

### 3. Market Up & Down Movement Classification
- Classifies markets into:
  - **Rising (तेजी)**: Price momentum $\ge +2.0\%$ with moderate volatility.
  - **Falling (मंदी)**: Price momentum $\le -2.0\%$.
  - **Stable (स्थिर)**: Steady price movement within $\pm 2.0\%$.
  - **Highly Volatile (अस्थिर)**: Volatility standard deviation $> 4.5$.

### 4. Farmer Profit & Cultivation ROI Calculator
- Takes quantity, selling price, hauling distance, freight rate, mandi cess, and cultivation cost.
- Outputs Gross Market Value, Total Deductions, Net Realization, Net Farming Profit, and Return on Investment (ROI %).

### 5. Government Minimum Support Price (MSP) Intelligence
- Built-in official Government MSP database for 2024-25 and 2025-26 seasons:
  - **Wheat:** ₹2,425/quintal
  - **Paddy (Common):** ₹2,300/quintal | **Paddy (Grade A):** ₹2,320/quintal
  - **Mustard:** ₹5,950/quintal
  - **Maize:** ₹2,225/quintal
  - **Soybean:** ₹4,892/quintal
  - **Cotton:** ₹7,121/quintal
  - **Gram (Chana):** ₹5,650/quintal
- Live comparison against current market rates with procurement advisory.

### 6. Radiant Pastel Design & Multilingual Localization
- **Radiant Pastel Palette:** Designed for outdoor sunlight readability using soft mint (`#E1F1E8`), sky blue (`#DCEBF5`), warm cream (`#FAF7F2`), and forest green (`#1E5128`). Corrupted icon ligatures and dark mode clashes are completely eliminated.
- **Multilingual Support:** Complete localization in:
  - **English**
  - **Hindi (हिंदी)**
  - **Punjabi (ਪੰਜਾਬੀ)**
- **Mobile-First Responsive Layout:** Large touch targets, intuitive cards, and clean typography.

### 7. Interactive Map & Market Comparison (🟢 Green & 🔵 Blue Marks)
- **Interactive Geospatial Visualization:** High-resolution PyDeck map plotting reachable mandis:
  - 🟢 **Green Marks (Top 3):** Maximum Net Realization hubs (highest in-pocket cash).
  - 🔵 **Blue Marks (Top 4–10):** High-performing secondary market hubs.
  - ⚪ **Grey Marks:** Other reachable mandis within radius.
  - ⭐ **Amber Star:** Farmer's selected origin/village location.
- **Click-to-Inspect & Quick-Pick Buttons:** Click any mark directly on the map or tap the 1-click quick-select pills to inspect the mandi's full profit breakdown.
- **Side-by-Side Comparison vs Local Mandi:** Shows exact extra cash (e.g. `+₹20,200 Extra Cash`) after accounting for vehicle freight, road distance, and mandi cess.
- **Market Switching Insights:** Explains dynamic changes when the farmer switches their origin market (GPS relocation, road distances recomputation, freight re-evaluation, and updated rankings).

### 8. Added Advanced Practical Decision Features
- 🚚 **Vehicle Freight Calculator:** Compare per-quintal transport costs across Tractor Trolley (₹1.80/q-km), Tata Ace (₹1.40/q-km), Bolero Pickup (₹1.10/q-km), and 10-Wheeler Truck (₹0.55/q-km).
- 🌾 **Moisture Deduction Estimator:** Calculates fair average quality (FAQ 12% moisture) dockage vs moisture content (14%–20%) and displays potential savings from sun-drying.
- 🤝 **FPO Group Pooling Savings:** Models 3–10 smallholder farmers pooling produce into a single 10-wheeler truck, saving thousands in freight expenses.
- ⚖️ **Break-Even Trader Offer Validator:** Calculates the exact minimum price an off-market local buyer must offer to beat traveling to the best APMC mandi.
- 📄 **Mandi Dispatch Gate Pass:** Generates a printable gate pass with farmer name, vehicle type, quantity, estimated net realization, and route details.

---

## Project Research & Intelligence (Preserved Academic Core)

All 4 original SIH26132 research objectives and 14 academic tabs remain completely intact:
- **Objective 1 (Data Processing & Features):**
  - Agmarknet 269k wheat records loader, invalid price repair, coverage filtering (60% calendar days), 15 engineered features, and correlation pruning ($|r| > 0.90$).
- **Objective 2 (Clustering & Architecture):**
  - Euclidean K-Means & Manhattan K-Medians across $K=2..10$.
  - Silhouette, Davies-Bouldin, and composite selection.
  - 2D PCA projection (explaining $>60\%$ variance).
  - Ward hierarchical linkage and cophenetic correlation.
- **Objective 3 (Validation & Stability):**
  - Cluster profiling and centroid heatmaps.
  - Eta-squared feature contribution analysis.
  - Stability validation across 20 random seeds, 50 bootstrap resamples ($ARI \approx 0.66$, moderately stable).
- **Objective 4 (Decision Support & Export):**
  - Academic advisor methodology.
  - 1-click Download package generating `results.zip` with all 27 300-DPI charts and CSV outputs.

---

## Quickstart & Run Instructions

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Launch MandiMitra
```bash
python -m streamlit run app.py
```

### 3. Open in Browser
Open `http://localhost:8501` to use the application on desktop, tablet, or smartphone.

---

## Directory Structure

```
MandiMitra/
├── app.py                      # Master Streamlit application (Farmer App + Research)
├── config.yaml                 # Pipeline and algorithm configuration
├── requirements.txt            # Python dependencies
├── test_smoke.py               # Comprehensive 8-point smoke test suite
├── src/
│   ├── advisor.py              # Distance ranking and net realization engine
│   ├── clustering.py           # K-Means, K-Medians, and Ward linkage
│   ├── commodity_manager.py    # Multi-commodity isolation & dataset manager
│   ├── farmer_components.py    # Mobile-first CSS, SVG sparklines, cards
│   ├── features.py             # Feature engineering & correlation pruning
│   ├── geocode.py              # Offline mandi gazetteer geocoding
│   ├── i18n.py                 # English, Hindi, and Punjabi dictionaries
│   ├── market_intelligence.py  # Decision engine, Sell/Wait, MSP, Profit calc
│   ├── pipeline.py             # Pipeline execution and figure generation
│   ├── preprocess.py           # Agmarknet cleaning, repair, and parser
│   ├── validation.py           # Stability, bootstrap ARI, and cluster metrics
│   └── visualise.py            # High-resolution figure plotting functions
├── data/
│   ├── agmarknet_wheat.csv.gz  # Real Agmarknet wheat dataset (269,406 rows)
│   └── commodities/            # Isolated stores per commodity
│       ├── Wheat/
│       ├── Rice (Paddy)/
│       ├── Mustard/
│       ├── Maize/
│       └── Potato/
└── outputs/                    # Precomputed wheat research outputs & figures
```

---

## Known Scientific Limitations
1. **Reporting Regularity:** Only mandis reporting on $\ge 60\%$ of days are clustered to avoid noise; low-volume rural mandis may be underrepresented.
2. **Quality & Grade Mixing:** Agmarknet modal prices aggregate across grades; farmers should inspect variety specs for high-premium quotes.
3. **Decision Support, Not Guaranteed Forecasts:** Future outlooks are conservative mathematical projections based on historical momentum and storage decay. They are not guaranteed prices.
