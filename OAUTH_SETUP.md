# 🔐 GitHub OAuth 2.0 Setup & Deployment

## 🎯 OAuth 2.0 Authentication for GitHub

OAuth 2.0 is the recommended authentication method for GitHub repositories. Here's how to set it up:

## 📋 Step-by-Step OAuth Setup

### 1️⃣ **Create Personal Access Token (OAuth 2.0)**

1. **Go to GitHub Settings**:
   - Navigate to: https://github.com/settings/tokens
   - Or: GitHub.com → Profile → Settings → Developer settings → Personal access tokens → Tokens (classic)

2. **Generate New Token**:
   - Click "Generate new token" → "Generate new token (classic)"
   - **Note**: Enter "Brazilian E-Commerce Analytics Platform"
   - **Expiration**: Choose "No expiration" or "90 days"
   - **Scopes**: Select these permissions:
     - ✅ `repo` (Full control of private repositories)
     - ✅ `workflow` (Update GitHub Action workflows)
     - ✅ `write:packages` (Upload packages)
     - ✅ `delete:packages` (Delete packages)

3. **Copy Token**:
   - Click "Generate token"
   - **⚠️ IMPORTANT**: Copy the token immediately (you won't see it again!)
   - Save it securely (e.g., password manager)

### 2️⃣ **Configure Git with OAuth Token**

```bash
# Set up git with your credentials
git config --global user.name "prsuman1"
git config --global user.email "your-email@example.com"  # Replace with your GitHub email

# Configure credential helper (stores token securely)
git config --global credential.helper store
```

### 3️⃣ **Deploy to Repository (OAuth Method)**

```bash
# Navigate to your project directory
cd "/Users/suman/Documents/Zeno Health part 1"

# Initialize git repository
git init

# Add remote repository  
git remote add origin https://github.com/prsuman1/Brazilian-E-Commerce-Public-Dataset-EDA.git

# Pull existing content to avoid conflicts
git pull origin main --allow-unrelated-histories

# Add all files
git add .

# Create commit with detailed message
git commit -m "🚀 Deploy Complete Brazilian E-Commerce Analytics Platform

✨ Features Added:
- Executive Dashboard with real-time KPIs
- Advanced Analytics (Revenue, Customer, Seller, Logistics)  
- Waterfall Charts & Sankey Diagrams
- Customer Cohort Analysis & Segmentation
- Geographic Bias-Adjusted Insights
- Strategic Recommendations with ROI Projections
- Comprehensive Technical Documentation

🔧 Technical Stack:
- Streamlit web application (110KB codebase)
- Plotly interactive visualizations
- Pandas data processing (128M+ records)
- Population-adjusted geographic analysis
- Data validation notebook included
- Docker & cloud deployment ready

📊 Business Impact:
- Analyzes 99,441 unique orders
- Covers 99,441+ customers across Brazil
- 3,095 seller performance analysis
- Strategic insights with 313% ROI projections

🎯 Ready for Production:
- One-command setup (streamlit run app.py)
- Professional BI-grade metrics & calculations
- Mobile responsive design
- Performance optimized with caching"

# Push to GitHub (will prompt for credentials)
git push -u origin main
```

## 🔑 **Authentication Process**

When you run `git push`, you'll be prompted:

```
Username for 'https://github.com': prsuman1
Password for 'https://prsuman1@github.com': [PASTE YOUR OAUTH TOKEN HERE]
```

**Important**: 
- Username: `prsuman1`
- Password: **Your OAuth token** (not your GitHub password!)

## 🔒 **Secure Token Storage**

After first use, Git will store your credentials securely. Future pushes won't require re-entering the token.

### Alternative: Store Token in Git Config
```bash
# Store token directly in git config (less secure but convenient)
git remote set-url origin https://prsuman1:YOUR_TOKEN_HERE@github.com/prsuman1/Brazilian-E-Commerce-Public-Dataset-EDA.git
```

## 🚀 **Ready-to-Execute Commands**

```bash
# Complete OAuth setup and deployment:

# 1. Configure Git (one-time setup)
git config --global user.name "prsuman1"
git config --global credential.helper store

# 2. Navigate to project
cd "/Users/suman/Documents/Zeno Health part 1"

# 3. Initialize and connect to GitHub
git init
git remote add origin https://github.com/prsuman1/Brazilian-E-Commerce-Public-Dataset-EDA.git

# 4. Sync with existing repository  
git pull origin main --allow-unrelated-histories

# 5. Add all project files
git add .

# 6. Create comprehensive commit
git commit -m "🚀 Complete Brazilian E-Commerce Analytics Platform

Professional BI dashboard with advanced visualizations:
- Executive KPI dashboard
- Revenue waterfall analysis  
- Customer cohort analysis
- Seller performance optimization
- Geographic bias-adjusted insights
- Strategic recommendations with ROI projections
- Comprehensive technical documentation

Built with Streamlit, Plotly, Pandas
Ready for production deployment"

# 7. Push to GitHub (OAuth authentication)
git push -u origin main
```

## ⚠️ **Troubleshooting OAuth Issues**

### Issue 1: Token Not Working
```bash
# Clear stored credentials and retry
git config --global --unset credential.helper
git config --global credential.helper store
# Then retry git push
```

### Issue 2: Repository Already Exists Error
```bash
# Force push (⚠️ Use carefully - replaces existing content)
git push -f origin main

# Or merge conflicts manually:
git pull origin main --no-rebase
# Resolve conflicts, then:
git add .
git commit -m "Merge conflicts resolved"  
git push origin main
```

### Issue 3: Large Files Error
```bash
# If CSV files too large, use Git LFS:
git lfs install
git lfs track "*.csv"
git add .gitattributes
git commit -m "Add Git LFS for large CSV files"
git push origin main
```

## ✅ **Verification Steps**

After successful push:

1. **Check Repository**: Visit https://github.com/prsuman1/Brazilian-E-Commerce-Public-Dataset-EDA
2. **Verify README**: Confirm new README.md displays properly
3. **Test Clone**: `git clone https://github.com/prsuman1/Brazilian-E-Commerce-Public-Dataset-EDA.git`
4. **Test Application**: `cd Brazilian-E-Commerce-Public-Dataset-EDA && pip install -r requirements.txt && streamlit run app.py`

## 🎉 **Success Indicators**

✅ Repository updated with new README  
✅ All application files uploaded  
✅ Professional documentation visible  
✅ Application runs successfully after clone  
✅ Streamlit app accessible at http://localhost:8501  

## 📞 **Next Steps**

Once pushed successfully:
1. **Repository will be updated** with comprehensive analytics platform
2. **Professional README** will be displayed on GitHub
3. **Others can clone and run** using the installation instructions
4. **Consider adding GitHub Actions** for automated testing
5. **Star the repository** to increase visibility

**Ready to proceed? Run the commands above with your OAuth token when prompted!**