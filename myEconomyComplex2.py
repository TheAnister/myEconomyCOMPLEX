#!/usr/bin/env python3
"""
UltraEconomy - Advanced Economic Simulator

A comprehensive economic simulation featuring:
- Individual-based economic modeling with detailed citizen simulation
- Complex business ecosystem with market interactions
- Multifaceted tax system with adjustable brackets and rates
- Government spending allocation across departments with real economic impacts
- Central bank monetary policy and inflation dynamics
- Financial market simulation with stock trading
- Housing market with potential bubbles and crash scenarios
- Modern Tesla-inspired dark mode UI with customizable dashboards
- Gemini API integration for complex economic events and AI-driven behaviors
- Detailed macro statistics tracking and data visualization

This simulation tool is designed for economic modeling, policy testing, and visualization.
"""

import sys
import random
import math
import time
import multiprocessing
import json
import os
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union

try:
    # Optional Gemini API integration for enhanced AI-driven events
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Modern UI packages
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSplashScreen, QProgressBar, QLabel, 
    QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QTabWidget, 
    QTableWidget, QTableWidgetItem, QHeaderView, QSpinBox, QComboBox,
    QFormLayout, QDoubleSpinBox, QInputDialog, QSlider, QGroupBox,
    QScrollArea, QSizePolicy, QFrame, QSplitter, QDialog,
    QCheckBox, QLineEdit, QTextEdit, QMenu, QStyle, QToolBar, QStatusBar,
    QToolButton, QRadioButton, QListWidget, QListWidgetItem,
    QStyledItemDelegate, QButtonGroup, QFileDialog, QMessageBox, QGridLayout
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QSize, QTimer, QSortFilterProxyModel, 
    QAbstractTableModel, QModelIndex, QPoint, QRect, QObject, QEvent,
    QPropertyAnimation, QEasingCurve, QMargins, QUrl, QDate
)
from PyQt6.QtGui import (
    QPixmap, QColor, QBrush, QPen, QFont, QIcon, QPainter, 
    QLinearGradient, QRadialGradient, QConicalGradient, QPalette, 
    QAction, QKeySequence, QCursor, QFontDatabase, QFontMetrics,
    QTransform, QMovie, QImage, QTextCursor, QPainterPath
)

# Data visualization
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.style.use('dark_background')

# -----------------------------
# Configuration and Constants
# -----------------------------
class Config:
    """Central configuration for the economic simulator"""
    # Simulation parameters
    DEFAULT_NUM_PERSONS = 100000 
    DEFAULT_NUM_COMPANIES = 1000
    SIMULATION_START_YEAR = 2025
    SIMULATION_START_MONTH = 1
    
    # Economic constants
    INFLATION_TARGET = 0.02  # 2% annual inflation target
    NATURAL_UNEMPLOYMENT = 0.04  # 4% natural unemployment rate
    AVG_PRODUCTIVITY_GROWTH = 0.015  # 1.5% annual productivity growth
    
    # UI settings
    UI_REFRESH_RATE = 1000  # ms
    DARK_MODE = True
    CHART_UPDATE_INTERVAL = 2  # Update charts every 2 months
    
    # Tax system defaults
    DEFAULT_TAX_BRACKETS = [
        (0, 12570, 0.0),       # Personal allowance
        (12570, 50270, 0.20),  # Basic rate
        (50270, 150000, 0.40), # Higher rate
        (150000, float('inf'), 0.45)  # Additional rate
    ]
    DEFAULT_CORPORATION_TAX = 0.25  # 25% corporation tax
    DEFAULT_SALES_TAX = 0.20  # 20% VAT/sales tax
    DEFAULT_NATIONAL_INSURANCE = 0.12  # 12% NI
    
    # Government spending defaults (as % of GDP)
    DEFAULT_GOVT_SPENDING = {
        "healthcare": 0.078,  # 7.8% of GDP
        "education": 0.042,   # 4.2% of GDP
        "welfare": 0.097,     # 9.7% of GDP
        "defense": 0.021,     # 2.1% of GDP
        "infrastructure": 0.032,  # 3.2% of GDP
        "research": 0.012,    # 1.2% of GDP
        "debt_interest": 0.018,  # 1.8% of GDP
        "public_services": 0.034,  # 3.4% of GDP
        "environment": 0.015,  # 1.5% of GDP
        "culture": 0.007      # 0.7% of GDP
    }
    
    # Colors for modern Tesla-like UI
    COLORS = {
        "background": "#121212",
        "surface": "#1E1E1E",
        "surface_light": "#252525",
        "primary": "#01A9F4",
        "secondary": "#6C757D",
        "success": "#28A745",
        "danger": "#DC3545",
        "warning": "#FFC107",
        "info": "#17A2B8",
        "text_primary": "#FFFFFF",
        "text_secondary": "#B0B0B0",
        "border": "#333333",
        "chart_grid": "#333333",
        "chart_line": "#01A9F4",
        "positive": "#4CAF50",
        "negative": "#F44336"
    }
    
    # Chart colors
    CHART_COLORS = [
        "#01A9F4", "#F44336", "#4CAF50", "#FFC107", 
        "#9C27B0", "#FF5722", "#3F51B5", "#E91E63",
        "#009688", "#673AB7", "#FFEB3B", "#2196F3"
    ]
    
    # Defaults for economy
    DEFAULT_INTEREST_RATE = 0.05  # 5% base interest rate
    DEFAULT_UNEMPLOYMENT = 0.055  # 5.5% initial unemployment
    DEFAULT_INFLATION = 0.023  # 2.3% initial inflation
    
    # Business sectors and types
    BUSINESS_SECTORS = [
        "Technology", "Manufacturing", "Retail", "Financial", 
        "Healthcare", "Energy", "Transport", "Agriculture",
        "Entertainment", "Construction", "Education", "Food"
    ]

# -----------------------------
# Helper Functions
# -----------------------------
def format_currency(n, symbol="£"):
    """Format a number as currency with appropriate scaling"""
    if n >= 1e12:
        return f"{symbol}{n/1e12:.2f}T"
    elif n >= 1e9:
        return f"{symbol}{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{symbol}{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{symbol}{n/1e3:.2f}K"
    else:
        return f"{symbol}{n:.2f}"

def format_percent(n):
    """Format a number as percentage"""
    return f"{n*100:.2f}%"

def clamp(value, min_val, max_val):
    """Clamp a value between min and max"""
    return max(min_val, min(max_val, value))

def weighted_random(weights_dict):
    """Select a random item from a dictionary based on weights"""
    options = list(weights_dict.keys())
    weights = list(weights_dict.values())
    return random.choices(options, weights=weights, k=1)[0]

def sigmoid(x):
    """Sigmoid function for smooth transitions"""
    return 1 / (1 + math.exp(-x))

def weighted_choice(choices):
    """Make a weighted random choice from list of (option, weight) tuples"""
    total = sum(weight for _, weight in choices)
    r = random.uniform(0, total)
    upto = 0
    for choice, weight in choices:
        upto += weight
        if upto > r:
            return choice
    return choices[-1][0]  # Fallback

def normal_clamp(mean, std_dev, min_val=0, max_val=1):
    """Generate a normally distributed random number clamped between values"""
    value = random.normalvariate(mean, std_dev)
    return clamp(value, min_val, max_val)

def log_normal(mean, sigma):
    """Generate a log-normal distributed value"""
    return math.exp(random.normalvariate(math.log(mean), sigma))

def generate_id():
    """Generate a unique ID"""
    return f"{int(time.time())}-{random.randint(10000, 99999)}"

# -----------------------------
# Person Class
# -----------------------------
class Person:
    def __init__(self, person_id):
        self.id = person_id
        self.name = self.generate_name()
        
        # Basic demographics
        self.age = random.randint(18, 90)
        self.gender = random.choice(["Male", "Female", "Non-binary"])
        self.education_level = self.determine_education_level()
        
        # Personality traits (0-1 scale)
        self.risk_tolerance = normal_clamp(0.5, 0.2)
        self.innovation = normal_clamp(0.5, 0.2)
        self.work_ethic = normal_clamp(0.6, 0.15)
        self.entrepreneurship = normal_clamp(0.3, 0.25)
        self.financial_literacy = normal_clamp(0.5, 0.2)
        
        # Economic attributes
        self.income = self.determine_starting_income()
        self.monthly_income = self.income / 12
        self.net_worth = self.determine_starting_net_worth()
        self.savings = self.net_worth * random.uniform(0.2, 0.5)
        self.savings_rate = random.uniform(0.05, 0.3)
        self.spending_rate = random.uniform(0.5, 0.9)
        self.tax_paid = 0
        self.disposable_income = 0
        
        # Employment
        self.employment_status = "Unemployed"
        self.employer = None
        self.job_title = None
        self.job_satisfaction = random.uniform(0.3, 0.8)
        self.years_experience = max(0, self.age - 18 - random.randint(0, 5)) 
        self.skill_level = self.calculate_skill_level()
        self.skills = self.generate_skills()
        self.productivity = normal_clamp(0.5, 0.15)
        self.years_experience = max(0, self.age - 18 - random.randint(0, 5))
        
        # Business ownership
        self.is_business_owner = False
        self.business = None
        
        # Housing
        self.housing_status = random.choices(
            ["Renter", "Owner", "Living with family"], 
            weights=[0.4, 0.5, 0.1]
        )[0]
        self.housing_value = 0 if self.housing_status != "Owner" else random.uniform(100000, 500000)
        self.housing_payment = 0
        
        # Assets
        self.investments = {}
        self.investment_value = 0
        self.stock_portfolio = {}
        
        # Social
        self.political_views = random.choices(
            ["Far Left", "Left", "Center", "Right", "Far Right"],
            weights=[0.1, 0.25, 0.3, 0.25, 0.1]
        )[0]
        self.social_mobility = random.uniform(0, 1)
        self.happiness = random.uniform(0.4, 0.8)
        self.health = normal_clamp(0.8, 0.15)
        
        # Tracking
        self.income_history = [self.monthly_income]
        self.net_worth_history = [self.net_worth]
        self.tax_history = [0]
        
        # External factors
        self.consumer_confidence = random.uniform(0.4, 0.8)
        
        # Life events
        self.life_events = []
        
    def generate_name(self):
        """Generate a random name"""
        first_names = [
            "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles",
            "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen"
        ]
        last_names = [
            "Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor",
            "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia", "Martinez", "Robinson"
        ]
        return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    def determine_education_level(self):
        """Determine education level based on age and probabilities"""
        education_levels = {
            "No Formal Education": 0.05,
            "Primary": 0.10,
            "Secondary": 0.35,
            "College": 0.25,
            "Bachelor's": 0.15,
            "Master's": 0.07,
            "Doctorate": 0.03
        }
        
        # Adjust probabilities based on age
        if self.age < 22:
            # Younger people less likely to have advanced degrees
            for level in ["Master's", "Doctorate"]:
                education_levels[level] *= 0.2
            education_levels["Bachelor's"] *= 0.5
            education_levels["Secondary"] *= 1.5
        elif self.age > 60:
            # Older generation less likely to have degrees
            education_levels["No Formal Education"] *= 2
            education_levels["Primary"] *= 1.5
            education_levels["Bachelor's"] *= 0.7
            education_levels["Master's"] *= 0.7
            
        return weighted_random(education_levels)
    
    def calculate_skill_level(self):
        """Calculate base skill level from education and age/experience"""
        education_values = {
            "No Formal Education": 0.1,
            "Primary": 0.2,
            "Secondary": 0.4,
            "College": 0.6,
            "Bachelor's": 0.7,
            "Master's": 0.85,
            "Doctorate": 0.95
        }
        
        base_skill = education_values.get(self.education_level, 0.5)
        experience_factor = min(1.0, self.years_experience / 40)  # Caps at 40 years experience
        
        # Skill level is a combination of education and experience
        skill_level = base_skill * 0.6 + experience_factor * 0.4
        
        # Add some randomness
        skill_level = normal_clamp(skill_level, 0.1)
        
        return skill_level
    
    def generate_skills(self):
        """Generate specific skills based on education level"""
        all_skills = {
            "technical": 0,
            "managerial": 0,
            "creative": 0,
            "analytical": 0,
            "communication": 0,
            "physical": 0,
            "financial": 0
        }
        
        # Base skill level affects all skills
        for skill in all_skills:
            all_skills[skill] = normal_clamp(self.skill_level * random.uniform(0.7, 1.3), 0.15)
        
        # Education level affects specific skills
        if self.education_level in ["Bachelor's", "Master's", "Doctorate"]:
            all_skills["analytical"] += random.uniform(0.1, 0.3)
            all_skills["technical"] += random.uniform(0.1, 0.3)
            
        if self.education_level in ["Master's", "Doctorate"]:
            all_skills["managerial"] += random.uniform(0.1, 0.2)
            
        # Cap at 1.0
        for skill in all_skills:
            all_skills[skill] = min(1.0, all_skills[skill])
            
        return all_skills
    
    def determine_starting_income(self):
        """Determine starting income based on education and age"""
        education_multipliers = {
            "No Formal Education": 0.5,
            "Primary": 0.6,
            "Secondary": 0.8,
            "College": 1.0,
            "Bachelor's": 1.3,
            "Master's": 1.6,
            "Doctorate": 2.0
        }
        
        # Base income range
        base_income = random.uniform(18000, 28000)
        
        # Apply education multiplier
        multiplier = education_multipliers.get(self.education_level, 1.0)
        income = base_income * multiplier
        
        # Apply age/experience curve - peaks around age 50-55
        age_factor = 1.0
        if self.age < 25:
            age_factor = 0.6 + (self.age - 18) * 0.05  # Gradual increase from age 18-25
        elif self.age < 55:
            age_factor = 0.9 + (self.age - 25) * 0.01  # Peak at age 55
        else:
            age_factor = 1.3 - (self.age - 55) * 0.01  # Gradual decrease after 55
        
        income *= age_factor
        
        # Add some randomness for individual variation
        income *= random.uniform(0.85, 1.15)
        
        return income
    
    def determine_starting_net_worth(self):
        """Determine starting net worth based on age and income"""
        # Base net worth is a function of income and age
        base_net_worth = self.income * 0.5
        
        # People accumulate wealth as they age
        age_factor = max(0, (self.age - 18) / 50)  # Normalized age factor
        
        # Apply age curve - wealth accelerates with age then plateaus
        age_multiplier = age_factor ** 2 * 15  # Squared for accelerating curve
        
        net_worth = base_net_worth * age_multiplier
        
        # Add some randomness - some people save more/less or have inheritance
        net_worth *= random.uniform(0.3, 3.0)
        
        # Ensure minimum net worth
        return max(1000, net_worth)
    
    def calculate_tax(self, tax_brackets):
        """Calculate income tax based on tax brackets"""
        annual_income = self.monthly_income * 12
        tax = 0
        
        for lower, upper, rate in tax_brackets:
            if annual_income > lower:
                taxable = min(annual_income, upper) - lower
                tax += taxable * rate
        
        # Convert to monthly tax
        monthly_tax = tax / 12
        self.tax_paid = monthly_tax
        return monthly_tax
    
    def calculate_disposable_income(self, tax_amount, national_insurance_rate):
        """Calculate disposable income after taxes"""
        ni_contribution = self.monthly_income * national_insurance_rate
        self.disposable_income = self.monthly_income - tax_amount - ni_contribution
        return self.disposable_income
    
    def make_economic_decisions(self, economy):
        """Make economic decisions based on disposable income"""
        # Savings decision
        adjusted_savings_rate = self.savings_rate
        
        # Adjust savings rate based on economic conditions
        if economy.global_conditions["inflation"] > 0.05:
            adjusted_savings_rate *= 1.2  # Save more during high inflation
        if economy.global_conditions["unemployment"] > 0.08:
            adjusted_savings_rate *= 1.3  # Save more during high unemployment
        if economy.central_bank.interest_rate > 0.07:
            adjusted_savings_rate *= 1.2  # Save more when interest rates are high
            
        # Adjust based on consumer confidence
        adjusted_savings_rate *= (2 - self.consumer_confidence)
        
        # Save money
        savings_amount = self.disposable_income * adjusted_savings_rate
        self.savings += savings_amount
        
        # Spending calculation (after savings)
        available_for_spending = self.disposable_income - savings_amount
        
        # Housing payment
        if self.housing_status == "Renter":
            self.housing_payment = available_for_spending * random.uniform(0.25, 0.4)
        elif self.housing_status == "Owner":
            self.housing_payment = available_for_spending * random.uniform(0.2, 0.35)
        else:  # Living with family
            self.housing_payment = available_for_spending * random.uniform(0.05, 0.15)
            
        # Remainder after housing
        remainder = available_for_spending - self.housing_payment
        
        # Investment decision
        investment_rate = self.financial_literacy * self.risk_tolerance * 0.2
        investment_amount = remainder * investment_rate
        
        # Add to investments
        self.investment_value += investment_amount
        
        # Consider starting a business
        self.consider_starting_business(economy)
        
        # Update net worth
        self.update_net_worth()
        
    def consider_starting_business(self, economy):
        """Consider starting a business based on personality and economic conditions"""
        if self.is_business_owner or self.savings < 25000:
            return False
            
        # Base probability from entrepreneurship trait
        base_probability = self.entrepreneurship * 0.01  # 0-1% monthly chance
        
        # Adjust based on economic conditions
        if economy.global_conditions["economic_growth"] > 0.02:
            base_probability *= 1.5  # More likely during good economic times
        elif economy.global_conditions["economic_growth"] < 0:
            base_probability *= 0.5  # Less likely during recession
            
        # Adjust based on interest rates
        if economy.central_bank.interest_rate < 0.03:
            base_probability *= 1.3  # More likely with low interest rates
        elif economy.central_bank.interest_rate > 0.08:
            base_probability *= 0.7  # Less likely with high interest rates
            
        # Age factor - people in prime age more likely to start business
        age_factor = 1.0
        if 25 <= self.age <= 45:
            age_factor = 1.5
        elif self.age > 60:
            age_factor = 0.5
            
        # Education factor
        education_factor = 1.0
        if self.education_level in ["Bachelor's", "Master's", "Doctorate"]:
            education_factor = 1.5
            
        # Final probability
        final_probability = base_probability * age_factor * education_factor
        
        # Make decision
        if random.random() < final_probability:
            # Start business
            business_type = random.choice(list(economy.business_types.keys()))
            sector = economy.business_types[business_type]['sector']
            name = f"{self.name}'s {sector} Business"
            
            # Create business with starting capital from savings
            starting_capital = min(self.savings * 0.7, 100000)
            self.savings -= starting_capital
            
            business = Business(name, sector, self.id, business_type, self.education_level)
            business.capital = starting_capital
            
            # Register as owner
            self.is_business_owner = True
            self.business = business
            self.employment_status = "Self-employed"
            self.employer = business
            
            # Add to economy
            economy.companies.append(business)
            return True
            
        return False
    
    def update_net_worth(self):
        """Update net worth calculation"""
        business_value = 0
        if self.is_business_owner and self.business:
            business_value = self.business.calculate_value()
            
        self.net_worth = (
            self.savings + 
            self.investment_value + 
            self.housing_value +
            business_value
        )
        
        self.net_worth_history.append(self.net_worth)
        
    def update_monthly(self, economy):
        """Monthly update for the person"""
        # Update income if employed
        if self.employment_status == "Employed" and self.employer:
            # Base income from employer's sector
            base_sector_income = economy.sector_avg_income.get(self.employer.sector, 2500) / 12
            
            # Adjust for skill level and productivity
            skill_factor = 0.7 + (self.skill_level * 0.6)
            productivity_factor = 0.8 + (self.productivity * 0.4)
            
            # Company success factor
            company_factor = 1.0
            if hasattr(self.employer, 'profitability'):
                if self.employer.profitability > 0.15:
                    company_factor = 1.2
                elif self.employer.profitability < 0:
                    company_factor = 0.9
                    
            # Calculate monthly income
            self.monthly_income = base_sector_income * skill_factor * productivity_factor * company_factor
            
            # Random variation
            self.monthly_income *= random.uniform(0.95, 1.05)
            
        elif self.employment_status == "Self-employed" and self.business:
            # Owner takes income based on business profit
            if self.business.profit > 0:
                self.monthly_income = min(self.business.profit * 0.3, self.business.profit * 0.8)
            else:
                # Minimal income during loss periods
                self.monthly_income = max(1000, self.monthly_income * 0.8)
                
        elif self.employment_status == "Unemployed":
            # Unemployment benefits - 20% of average wage
            self.monthly_income = economy.avg_income * 0.2 / 12
            
            # Look for job
            self.seek_employment(economy)
            
        # Calculate taxes
        tax_amount = self.calculate_tax(economy.government.income_tax_brackets)
        
        # Update disposable income
        self.calculate_disposable_income(tax_amount, economy.government.national_insurance_rate)
        
        # Make economic decisions
        self.make_economic_decisions(economy)
        
        # Update consumer confidence based on personal and economy-wide factors
        self.update_consumer_confidence(economy)
        
        # Random life events
        self.process_life_events(economy)
        
        # Update other attributes
        self.productivity = normal_clamp(self.productivity, 0.05, 0.7, 1.0)  # Slight random variation
        self.job_satisfaction = normal_clamp(self.job_satisfaction, 0.1, 0.2, 1.0)
        
        # Track history
        self.income_history.append(self.monthly_income)
        self.tax_history.append(tax_amount)
        
    def update_consumer_confidence(self, economy):
        """Update consumer confidence based on economic conditions"""
        # Start with previous confidence
        new_confidence = self.consumer_confidence
        
        # Personal factors
        if len(self.income_history) > 1 and self.monthly_income > self.income_history[-2]:
            new_confidence += 0.02
        else:
            new_confidence -= 0.02
            
        if self.employment_status == "Unemployed":
            new_confidence -= 0.1
            
        # Economic factors
        if economy.global_conditions["economic_growth"] > 0.01:
            new_confidence += 0.01
        elif economy.global_conditions["economic_growth"] < 0:
            new_confidence -= 0.03
            
        if economy.global_conditions["inflation"] > 0.04:
            new_confidence -= 0.02
            
        if economy.global_conditions["unemployment"] > 0.07:
            new_confidence -= 0.01
            
        # Apply changes with smoothing
        self.consumer_confidence = clamp(
            self.consumer_confidence * 0.8 + new_confidence * 0.2,
            0.1, 0.9
        )
        
    def seek_employment(self, economy):
        """Look for employment if unemployed"""
        if self.employment_status != "Unemployed":
            return
            
        # Base chance of finding employment
        base_chance = 0.10  # 10% monthly chance
        
        # Adjust based on economic conditions
        if economy.global_conditions["unemployment"] > 0.08:
            base_chance *= 0.7
        elif economy.global_conditions["unemployment"] < 0.04:
            base_chance *= 1.5
            
        # Adjust based on education and skills
        education_factor = {
            "No Formal Education": 0.6,
            "Primary": 0.7,
            "Secondary": 0.9,
            "College": 1.1,
            "Bachelor's": 1.3,
            "Master's": 1.5,
            "Doctorate": 1.7
        }.get(self.education_level, 1.0)
        
        skill_factor = 0.7 + (self.skill_level * 0.6)
        
        # Final chance
        final_chance = base_chance * education_factor * skill_factor
        
        # Make decision
        if random.random() < final_chance:
            # Choose a random company that's hiring
            hiring_companies = [c for c in economy.companies if c.is_hiring()]
            if hiring_companies:
                company = random.choice(hiring_companies)
                self.employer = company
                self.employment_status = "Employed"
                self.job_title = company.generate_job_title(self.skill_level)
                company.hire_employee(self)
        
    def process_life_events(self, economy):
        """Process random life events"""
        # 2% chance of a significant life event each month
        if random.random() < 0.02:
            event_types = {
                "promotion": 0.2,
                "job_loss": 0.15,
                "health_issue": 0.15,
                "windfall": 0.1,
                "education": 0.1,
                "relocation": 0.15,
                "business_venture": 0.15
            }
            
            event_type = weighted_random(event_types)
            
            if event_type == "promotion" and self.employment_status == "Employed":
                # Promotion increases income and job satisfaction
                self.monthly_income *= random.uniform(1.05, 1.2)
                self.job_satisfaction += random.uniform(0.05, 0.15)
                self.life_events.append(("Promotion", economy.current_date))
                
            elif event_type == "job_loss" and self.employment_status == "Employed":
                # Chance of job loss
                if random.random() < 0.3:  # 30% chance the job loss happens
                    self.employment_status = "Unemployed"
                    self.employer = None
                    self.job_title = None
                    self.job_satisfaction = 0.5
                    self.life_events.append(("Job Loss", economy.current_date))
                
            elif event_type == "health_issue":
                # Health issue impacts productivity and potentially income
                severity = random.uniform(0.1, 0.5)
                self.health -= severity
                if self.health < 0.3:
                    self.productivity *= (0.7 + self.health)
                    if self.employment_status == "Employed":
                        self.monthly_income *= (0.8 + self.health * 0.2)
                self.life_events.append(("Health Issue", economy.current_date))
                
            elif event_type == "windfall":
                # Random financial windfall
                amount = self.monthly_income * random.uniform(1, 12)
                self.savings += amount
                self.life_events.append(("Financial Windfall", economy.current_date))
                
            elif event_type == "education" and random.random() < 0.3:
                # Further education that improves skills
                old_education = self.education_level
                education_levels = ["No Formal Education", "Primary", "Secondary", 
                                   "College", "Bachelor's", "Master's", "Doctorate"]
                
                current_idx = education_levels.index(self.education_level)
                if current_idx < len(education_levels) - 1:
                    # 30% chance to advance education level
                    self.education_level = education_levels[current_idx + 1]
                    self.skill_level = min(1.0, self.skill_level + random.uniform(0.05, 0.15))
                    self.financial_literacy += random.uniform(0.02, 0.08)
                    self.life_events.append((f"Education: {old_education} → {self.education_level}", 
                                            economy.current_date))
                
            elif event_type == "relocation":
                # Relocation affects housing
                if self.housing_status == "Renter":
                    # 50% chance to buy if they can afford it
                    if self.savings > 50000 and random.random() < 0.5:
                        self.housing_status = "Owner"
                        house_price = random.uniform(150000, 500000)
                        self.housing_value = house_price
                        self.savings -= house_price * 0.2  # 20% down payment
                        self.life_events.append(("Bought House", economy.current_date))
                elif self.housing_status == "Owner" and random.random() < 0.3:
                    # 30% chance to sell and upgrade
                    old_value = self.housing_value
                    sale_value = old_value * random.uniform(0.9, 1.5)
                    self.savings += sale_value
                    self.housing_value = sale_value * random.uniform(1.1, 1.5)
                    self.savings -= self.housing_value * 0.2  # 20% down payment
                    self.life_events.append(("Housing Upgrade", economy.current_date))
                
            elif event_type == "business_venture" and not self.is_business_owner:
                # Consider starting business
                if self.savings > 50000 and self.entrepreneurship > 0.6 and random.random() < 0.3:
                    self.consider_starting_business(economy)
                    if self.is_business_owner:
                        self.life_events.append(("Started Business", economy.current_date))

    def get_summary_stats(self):
        """Get summary statistics for display"""
        return {
            "name": self.name,
            "age": self.age,
            "education": self.education_level,
            "monthly_income": self.monthly_income,
            "net_worth": self.net_worth,
            "employment": self.employment_status,
            "job_title": self.job_title if self.job_title else "N/A",
            "employer": self.employer.name if self.employer else "N/A",
            "housing": self.housing_status,
            "savings": self.savings,
            "happiness": self.happiness,
            "health": self.health
        }

# -----------------------------
# Business Class
# -----------------------------
class Business:
    def __init__(self, name, sector, owner_id, business_type, education_level):
        self.id = generate_id()
        self.name = name
        self.sector = sector
        self.business_type = business_type
        self.owner_id = owner_id
        
        # Business financials
        self.revenue = self.determine_starting_revenue()
        self.costs = self.determine_starting_costs()
        self.profit = self.revenue - self.costs
        self.capital = random.uniform(50000, 200000)
        self.value = self.capital
        self.founding_date = datetime.now()
        self.age = 0  # in months
        
        # Business metrics
        self.employees = []
        self.customer_satisfaction = random.uniform(0.4, 0.8)
        self.market_share = 0
        self.innovation_level = random.uniform(0.3, 0.7)
        self.efficiency = random.uniform(0.4, 0.8)
        self.growth_rate = 0
        self.profitability = 0
        
        # Risk factors
        self.risk_level = random.uniform(0.2, 0.8)
        self.credit_rating = random.uniform(0.5, 0.9)
        
        # Technology and innovation
        self.technology_level = random.uniform(0.3, 0.8)
        self.r_and_d_investment = 0
        
        # Hiring
        self.hiring_budget = 0
        self.num_job_openings = random.randint(1, 10)
        self.max_employees = random.randint(10, 100)
        
        # Market and products
        self.product_price = random.uniform(10, 1000)
        self.marketing_budget = 0
        
        # Set initial values based on owner education
        self.adjust_initial_values(education_level)
        
        # History tracking
        self.revenue_history = [self.revenue]
        self.profit_history = [self.profit]
        self.employee_count_history = [len(self.employees)]
        self.value_history = [self.value]
        
        # Tax information
        self.tax_paid = 0
        self.tax_rate = 0
        
        # Stock market data
        self.is_public = random.random() < 0.10  # 10% chance to be public initially
        self.share_price = 0
        self.shares_outstanding = 1000000
        self.share_price_history = []
        
        if self.is_public:
            initial_valuation = max(self.capital * 2, self.revenue * 10)
            self.share_price = initial_valuation / self.shares_outstanding
            self.share_price_history = [self.share_price]
            
    def determine_starting_revenue(self):
        """Determine starting revenue for new business"""
        base_range = (50000, 500000)
        
        # Sector adjustments
        sector_multipliers = {
            "Technology": 1.2,
            "Financial": 1.3,
            "Healthcare": 1.15,
            "Energy": 1.4,
            "Retail": 0.9,
            "Manufacturing": 1.1,
            "Food": 0.85,
            "Construction": 1.05,
            "Entertainment": 0.95,
            "Transport": 1.0,
            "Agriculture": 0.8,
            "Education": 0.75
        }
        
        multiplier = sector_multipliers.get(self.sector, 1.0)
        
        # Apply randomization
        base = random.uniform(*base_range) * multiplier
        
        # Annual revenue, convert to monthly
        return base / 12
        
    def determine_starting_costs(self):
        """Determine starting costs based on revenue"""
        # Costs are a percentage of revenue, varying by sector
        cost_margins = {
            "Technology": (0.55, 0.75),
            "Financial": (0.60, 0.80),
            "Healthcare": (0.65, 0.85),
            "Energy": (0.70, 0.85),
            "Retail": (0.75, 0.90),
            "Manufacturing": (0.70, 0.85),
            "Food": (0.75, 0.90),
            "Construction": (0.70, 0.85),
            "Entertainment": (0.65, 0.85),
            "Transport": (0.75, 0.90),
            "Agriculture": (0.70, 0.85),
            "Education": (0.80, 0.95)
        }
        
        margin_range = cost_margins.get(self.sector, (0.70, 0.85))
        cost_ratio = random.uniform(*margin_range)
        
        return self.revenue * cost_ratio

    def adjust_initial_values(self, education_level):
        """Adjust initial business values based on owner education"""
        education_bonus = {
            "No Formal Education": 0,
            "Primary": 0.05,
            "Secondary": 0.1,
            "College": 0.15,
            "Bachelor's": 0.2,
            "Master's": 0.25,
            "Doctorate": 0.3
        }.get(education_level, 0.1)
        
        self.innovation_level += education_bonus
        self.efficiency += education_bonus
        self.technology_level += education_bonus
        self.customer_satisfaction += education_bonus * 0.5
        
        # Cap values at 1.0
        for attr in ['innovation_level', 'efficiency', 'technology_level', 'customer_satisfaction']:
            setattr(self, attr, min(1.0, getattr(self, attr)))

    def update_monthly(self, economy):
        """Monthly update for business operations"""
        # Increase age
        self.age += 1
        
        # Determine revenue based on sector and business performance
        sector_factor = economy.sector_performance.get(self.sector, 1.0)
        
        # Base revenue scales with employee count and business age
        employee_count = len(self.employees) + 1  # +1 for owner
        age_factor = min(1.0, self.age / 24)  # Businesses mature over 2 years
        
        # Base monthly revenue calculation
        base_revenue = 15000 * (1 + employee_count * 0.5) * age_factor
        
        # Adjust revenue based on business quality factors
        quality_factor = (
            self.customer_satisfaction * 0.3 +
            self.innovation_level * 0.2 +
            self.efficiency * 0.3 +
            self.technology_level * 0.2
        )
        
        # Economy-wide factors
        economic_factor = 1.0
        if economy.global_conditions["economic_growth"] > 0.02:
            economic_factor = 1.1
        elif economy.global_conditions["economic_growth"] < 0:
            economic_factor = 0.9
            
        # Consumer confidence affects businesses differently by sector
        consumer_confidence_effect = 1.0
        if self.sector in ["Retail", "Entertainment", "Food"]:
            # These sectors more affected by consumer confidence
            consumer_confidence = economy.global_conditions["consumer_confidence"]
            consumer_confidence_effect = 0.8 + (consumer_confidence * 0.4)
            
        # Calculate revenue
        self.revenue = (
            base_revenue * 
            sector_factor * 
            (0.7 + quality_factor * 0.6) * 
            economic_factor *
            consumer_confidence_effect
        )
        
        # Add randomness
        self.revenue *= random.uniform(0.9, 1.1)
        
        # Calculate costs
        employee_costs = sum(e.monthly_income for e in self.employees) if self.employees else 0
        fixed_costs = self.revenue * random.uniform(0.3, 0.5)  # Rent, utilities, etc.
        variable_costs = self.revenue * random.uniform(0.2, 0.4)  # Supplies, materials
        
        self.costs = employee_costs + fixed_costs + variable_costs + self.r_and_d_investment
        
        # Calculate profit before tax
        self.profit = self.revenue - self.costs
        
        # Pay corporate tax
        tax_amount = max(0, self.profit * economy.government.corporation_tax_rate)
        self.tax_paid = tax_amount
        economy.government.collect_tax(tax_amount, "corporate")
        
        # Profit after tax
        self.profit -= tax_amount
        
        # Update business metrics
        self.update_business_metrics()
        
        # Make business decisions
        self.make_business_decisions(economy)
        
        # Update capital based on profit
        if self.profit > 0:
            # Reinvest some profits, distribute some to owner
            reinvestment_rate = random.uniform(0.4, 0.7)
            self.capital += self.profit * reinvestment_rate
        else:
            # Losses reduce capital
            self.capital += self.profit
            
        # Update business value
        self.value = self.calculate_value()
        
        # Consider hiring or firing
        self.consider_hiring_firing(economy)
        
        # Consider IPO
        self.consider_ipo(economy)
        
        # Update share price if public
        if self.is_public:
            self.update_share_price(economy)
        
        # Track history
        self.revenue_history.append(self.revenue)
        self.profit_history.append(self.profit)
        self.employee_count_history.append(len(self.employees))
        self.value_history.append(self.value)
        
    def update_business_metrics(self):
        """Update key business metrics"""
        # Calculate profitability
        self.profitability = self.profit / self.revenue if self.revenue > 0 else 0
        
        # Calculate growth rate
        if len(self.revenue_history) > 3:
            prev_avg = sum(self.revenue_history[-4:-1]) / 3
            current = self.revenue
            self.growth_rate = (current - prev_avg) / prev_avg if prev_avg > 0 else 0
        else:
            self.growth_rate = 0
            
        # Update efficiency based on technology level and employee productivity
        avg_employee_productivity = 0
        if self.employees:
            avg_employee_productivity = sum(e.productivity for e in self.employees) / len(self.employees)
            
        self.efficiency = self.efficiency * 0.8 + (self.technology_level * 0.1 + avg_employee_productivity * 0.1)
        
        # Cap efficiency
        self.efficiency = clamp(self.efficiency, 0.2, 0.95)
        
    def make_business_decisions(self, economy):
        """Make key business decisions based on performance and economy"""
        # R&D investment decision
        if self.profit > 0 and self.capital > 10000:
            r_and_d_rate = self.innovation_level * random.uniform(0.05, 0.15)
            self.r_and_d_investment = self.profit * r_and_d_rate
            
            # R&D improves technology level
            tech_improvement = self.r_and_d_investment / (self.capital * 10) # Diminishing returns
            self.technology_level = min(0.95, self.technology_level + tech_improvement)
            
        else:
            self.r_and_d_investment = 0
            
        # Hiring budget based on growth and profitability
        if self.growth_rate > 0.05 and self.profitability > 0.1:
            # Growing and profitable - aggressive hiring
            self.hiring_budget = self.profit * random.uniform(0.1, 0.2)
            self.num_job_openings = max(1, int(self.hiring_budget / (economy.avg_income / 10)))
        elif self.growth_rate > 0 and self.profitability > 0:
            # Stable growth - moderate hiring
            self.hiring_budget = self.profit * random.uniform(0.05, 0.1)
            self.num_job_openings = max(0, int(self.hiring_budget / (economy.avg_income / 10)))
        else:
            # No growth or profit - no hiring
            self.hiring_budget = 0
            self.num_job_openings = 0
            
        # Marketing decisions affect customer satisfaction
        if self.profit > 0:
            marketing_spend = self.profit * random.uniform(0.05, 0.15)
            customer_satisfaction_boost = marketing_spend / (self.revenue * 2)
            self.customer_satisfaction = min(0.95, self.customer_satisfaction + customer_satisfaction_boost)
        else:
            # Customer satisfaction slowly decreases without investment
            self.customer_satisfaction = max(0.3, self.customer_satisfaction * 0.98)
            
    def consider_hiring_firing(self, economy):
        """Consider hiring or firing employees"""
        # Hiring
        if self.num_job_openings > 0:
            # Company is looking to hire
            pass  # Actual hiring happens in Person.seek_employment()
            
        # Firing
        elif self.profit < 0 and random.random() < 0.3:
            # Company not doing well, consider layoffs
            layoff_percent = random.uniform(0.1, 0.3)
            num_to_layoff = max(1, int(len(self.employees) * layoff_percent))
            
            for _ in range(num_to_layoff):
                if self.employees:
                    employee = random.choice(self.employees)
                    self.fire_employee(employee)
                    
    def consider_ipo(self, economy):
        """Consider going public with an IPO"""
        if self.is_public or self.age < 24:  # Must be at least 2 years old
            return
            
        # Requirements for IPO
        min_value = 5000000  # Minimum company value for IPO
        min_profit_streak = 4  # Minimum consecutive profitable months
        
        if self.value < min_value:
            return
            
        # Check profit streak
        profit_streak = 0
        for profit in reversed(self.profit_history):
            if profit > 0:
                profit_streak += 1
            else:
                break
                
        if profit_streak < min_profit_streak:
            return
            
        # Market conditions affect IPO decision
        market_factor = economy.stock_market.market_sentiment
        if market_factor < 0.4:  # Bad market conditions
            return
            
        # Calculate IPO probability
        base_probability = 0.05  # 5% monthly chance if requirements met
        adjusted_probability = base_probability * market_factor
        
        if random.random() < adjusted_probability:
            self.launch_ipo(economy)
            
    def launch_ipo(self, economy):
        """Launch an Initial Public Offering"""
        self.is_public = True
        self.shares_outstanding = 1000000
        
        # Initial share price based on valuation
        ipo_valuation = self.value * random.uniform(1.1, 1.5)  # IPO premium
        self.share_price = ipo_valuation / self.shares_outstanding
        self.share_price_history = [self.share_price]
        
        # Add to stock market
        economy.stock_market.add_public_company(self)
        
    def update_share_price(self, economy):
        """Update share price for public companies"""
        if not self.is_public:
            return
            
        # Base factors affecting share price
        profit_factor = 1.0
        growth_factor = 1.0
        market_factor = 1.0
        
        # Profit performance
        if self.profitability > 0.15:
            profit_factor = 1.1
        elif self.profitability > 0.05:
            profit_factor = 1.05
        elif self.profitability < 0:
            profit_factor = 0.9
            
        # Growth performance
        if self.growth_rate > 0.1:
            growth_factor = 1.15
        elif self.growth_rate > 0.05:
            growth_factor = 1.08
        elif self.growth_rate < 0:
            growth_factor = 0.92
            
        # Overall market performance
        market_sentiment = economy.stock_market.market_sentiment
        market_factor = 0.9 + (market_sentiment * 0.2)
        
        # Random factor (market noise)
        random_factor = random.uniform(0.95, 1.05)
        
        # Calculate price change
        price_change = profit_factor * growth_factor * market_factor * random_factor
        
        # Update share price
        self.share_price *= price_change
        self.share_price_history.append(self.share_price)
            
    def hire_employee(self, person):
        """Hire a new employee"""
        if person not in self.employees and len(self.employees) < self.max_employees:
            self.employees.append(person)
            self.num_job_openings = max(0, self.num_job_openings - 1)
            return True
        return False
            
    def fire_employee(self, person):
        """Fire an employee"""
        if person in self.employees:
            self.employees.remove(person)
            person.employment_status = "Unemployed"
            person.employer = None
            person.job_title = None
            person.job_satisfaction -= 0.2
            return True
        return False
            
    def is_hiring(self):
        """Check if company is hiring"""
        return self.num_job_openings > 0 and len(self.employees) < self.max_employees
        
    def calculate_value(self):
        """Calculate the value of the business"""
        # Business valuation based on assets, profit, and growth
        if len(self.profit_history) > 0:
            avg_annual_profit = sum(self.profit_history[-12:]) * 12 if len(self.profit_history) >= 12 else sum(self.profit_history) * (12 / len(self.profit_history))
            
            # Price-to-earnings multiplier based on growth and sector
            pe_ratio = 10  # Base P/E ratio
            
            if self.growth_rate > 0.1:
                pe_ratio += 10  # High growth premium
            elif self.growth_rate > 0.05:
                pe_ratio += 5   # Moderate growth premium
                
            # Profitability affects valuation
            if self.profitability > 0.2:
                pe_ratio += 3
                
            # Value based on earnings and capital
            profit_based_value = avg_annual_profit * pe_ratio
            asset_value = self.capital
            
            # Weighted average favoring profit-based value for established companies
            age_weight = min(0.8, self.age / 36)  # Increases up to 80% over 3 years
            value = profit_based_value * age_weight + asset_value * (1 - age_weight)
            
            # Minimum value is capital
            return max(asset_value, value)
        else:
            # New business is valued at its capital
            return self.capital
            
    def generate_job_title(self, skill_level):
        """Generate a job title based on skill level"""
        entry_jobs = ["Assistant", "Clerk", "Trainee", "Junior Associate"]
        mid_jobs = ["Associate", "Specialist", "Analyst", "Coordinator"]
        senior_jobs = ["Manager", "Director", "Senior Specialist", "Team Lead"]
        executive_jobs = ["Executive", "Head of Department", "Chief Officer", "VP"]
        
        sector_prefix = {
            "Technology": ["Software", "IT", "Technical", "Digital"],
            "Manufacturing": ["Production", "Assembly", "Quality", "Operations"],
            "Retail": ["Sales", "Customer", "Merchandising", "Retail"],
            "Financial": ["Financial", "Accounting", "Banking", "Investment"],
            "Healthcare": ["Medical", "Health", "Clinical", "Patient"],
            "Energy": ["Energy", "Power", "Utility", "Resource"],
            "Transport": ["Logistics", "Transport", "Fleet", "Shipping"],
            "Agriculture": ["Agricultural", "Farm", "Crop", "Livestock"],
            "Entertainment": ["Media", "Entertainment", "Creative", "Content"],
            "Construction": ["Construction", "Building", "Project", "Site"],
            "Education": ["Education", "Teaching", "Training", "Learning"],
            "Food": ["Food", "Culinary", "Kitchen", "Service"]
        }
        
        prefix = random.choice(sector_prefix.get(self.sector, ["General"]))
        
        if skill_level < 0.3:
            return f"{prefix} {random.choice(entry_jobs)}"
        elif skill_level < 0.6:
            return f"{prefix} {random.choice(mid_jobs)}"
        elif skill_level < 0.85:
            return f"{prefix} {random.choice(senior_jobs)}"
        else:
            return f"{prefix} {random.choice(executive_jobs)}"
            
    def get_summary_stats(self):
        """Get summary statistics for display"""
        return {
            "name": self.name,
            "sector": self.sector,
            "age": f"{self.age} months",
            "employees": len(self.employees),
            "revenue": self.revenue,
            "profit": self.profit,
            "profitability": f"{self.profitability*100:.1f}%",
            "growth": f"{self.growth_rate*100:.1f}%",
            "value": self.value,
            "technology": f"{self.technology_level*100:.1f}%",
            "satisfaction": f"{self.customer_satisfaction*100:.1f}%",
            "public": "Yes" if self.is_public else "No",
            "share_price": self.share_price if self.is_public else "N/A"
        }

# -----------------------------
# Government Class
# -----------------------------
class Government:
    def __init__(self):
        self.treasury = 1000000000  # Starting treasury balance
        self.tax_revenue = 0
        self.spending = 0
        self.budget_balance = 0  # Surplus or deficit
        self.debt = 5000000000  # Starting government debt
        self.debt_to_gdp = 0.7  # Initial debt-to-GDP ratio
        
        # Tax system
        self.income_tax_brackets = Config.DEFAULT_TAX_BRACKETS.copy()
        self.corporation_tax_rate = Config.DEFAULT_CORPORATION_TAX
        self.sales_tax_rate = Config.DEFAULT_SALES_TAX
        self.national_insurance_rate = Config.DEFAULT_NATIONAL_INSURANCE
        
        # Tax collections
        self.income_tax_collected = 0
        self.corporation_tax_collected = 0
        self.sales_tax_collected = 0
        self.national_insurance_collected = 0
        
        # Spending allocation (as proportion of total spending)
        self.spending_allocation = Config.DEFAULT_GOVT_SPENDING.copy()
        
        # Spending multipliers (effect on economy)
        self.spending_multipliers = {
            "healthcare": 1.4,
            "education": 1.6,
            "welfare": 1.3,
            "defense": 1.1,
            "infrastructure": 1.7,
            "research": 1.8,
            "debt_interest": 0.8,
            "public_services": 1.2,
            "environment": 1.5,
            "culture": 1.3
        }
        
        # Tracking
        self.revenue_history = []
        self.spending_history = []
        self.debt_history = []
        self.deficit_history = []
        
    def collect_tax(self, amount, tax_type):
        """Collect tax of a specific type"""
        self.treasury += amount
        self.tax_revenue += amount
        
        if tax_type == "income":
            self.income_tax_collected += amount
        elif tax_type == "corporate":
            self.corporation_tax_collected += amount
        elif tax_type == "sales":
            self.sales_tax_collected += amount
        elif tax_type == "national_insurance":
            self.national_insurance_collected += amount
            
    def spend(self, amount):
        """Government spending"""
        if amount <= self.treasury:
            self.treasury -= amount
            self.spending += amount
            return True
        else:
            # Need to borrow
            self.borrow(amount - self.treasury)
            self.treasury = 0
            self.spending += amount
            return True
            
    def borrow(self, amount):
        """Borrow money by issuing debt"""
        self.debt += amount
        self.treasury += amount
        
    def pay_debt_interest(self, interest_rate):
        """Pay interest on government debt"""
        # Calculate annual interest rate, pay monthly portion
        monthly_interest = self.debt * interest_rate / 12
        self.spend(monthly_interest)
        
    def update_spending_allocation(self, new_allocation):
        """Update spending allocation percentages"""
        # Validate that percentages add up to approximately 1
        if 0.98 <= sum(new_allocation.values()) <= 1.02:
            self.spending_allocation = new_allocation
            return True
        return False
        
    def update_tax_brackets(self, new_brackets):
        """Update income tax brackets"""
        self.income_tax_brackets = new_brackets
        
    def update_monthly(self, economy):
        """Monthly government update"""
        try:
            # Reset monthly counters
            self.tax_revenue = 0
            self.spending = 0
            self.income_tax_collected = 0
            self.corporation_tax_collected = 0
            self.sales_tax_collected = 0
            self.national_insurance_collected = 0
            
            # Calculate GDP-based spending
            target_spending = economy.gdp * sum(self.spending_allocation.values())
            monthly_target = target_spending / 12
            
            # Allocate spending by department
            total_multiplier_effect = 0
            for dept, proportion in self.spending_allocation.items():
                dept_spending = monthly_target * proportion
                self.spend(dept_spending)
                
                # Calculate economic effect of spending
                multiplier = self.spending_multipliers.get(dept, 1.0)
                effect = dept_spending * multiplier
                total_multiplier_effect += effect
                
            # Economic effects of government spending
            if getattr(economy, 'gdp', 0) > 0:
                economy.government_spending_effect = total_multiplier_effect / economy.gdp
            else:
                economy.government_spending_effect = 0.03  # Default effect if GDP is zero
            
            # Calculate debt to GDP ratio
            if getattr(economy, 'gdp', 0) > 0:
                self.debt_to_gdp = self.debt / economy.gdp
            else:
                self.debt_to_gdp = 1.0  # Default ratio if GDP is zero
            
            # Calculate budget balance (surplus/deficit)
            self.budget_balance = self.tax_revenue - self.spending
            
            # Pay debt interest
            self.pay_debt_interest(economy.central_bank.interest_rate)
            
            # Track history
            self.revenue_history.append(self.tax_revenue)
            self.spending_history.append(self.spending)
            self.debt_history.append(self.debt)
            self.deficit_history.append(self.budget_balance)
        except Exception as e:
            print(f"Government update error: {e}")
            
        
    def get_summary_stats(self):
        """Get summary statistics for display"""
        return {
            "treasury": self.treasury,
            "tax_revenue": self.tax_revenue * 12,  # Annualized
            "spending": self.spending * 12,  # Annualized
            "budget_balance": self.budget_balance * 12,  # Annualized
            "debt": self.debt,
            "debt_to_gdp": self.debt_to_gdp,
            "income_tax": self.income_tax_collected * 12,  # Annualized
            "corporation_tax": self.corporation_tax_collected * 12,  # Annualized
            "sales_tax": self.sales_tax_collected * 12,  # Annualized
            "national_insurance": self.national_insurance_collected * 12  # Annualized
        }

# -----------------------------
# Central Bank Class
# -----------------------------
class CentralBank:
    def __init__(self):
        self.interest_rate = Config.DEFAULT_INTEREST_RATE
        self.inflation_target = Config.INFLATION_TARGET
        self.money_supply = 10000000000  # Initial money supply
        self.reserve_ratio = 0.1  # Required reserve ratio for commercial banks
        
        # Monetary policy stance
        self.policy_stance = "Neutral"  # "Expansionary", "Neutral", or "Contractionary"
        
        # Tracking
        self.interest_rate_history = [self.interest_rate]
        self.inflation_history = [Config.DEFAULT_INFLATION]
        self.money_supply_history = [self.money_supply]
        
    def set_interest_rate(self, new_rate):
        """Set a new interest rate"""
        self.interest_rate = clamp(new_rate, 0.001, 0.20)  # Between 0.1% and 20%
        
    def adjust_policy(self, inflation, unemployment, economic_growth):
        """Adjust monetary policy based on economic conditions"""
        # Simple Taylor rule-inspired policy
        inflation_gap = inflation - self.inflation_target
        output_gap = Config.NATURAL_UNEMPLOYMENT - unemployment
        
        # Calculate desired interest rate change
        rate_change = (inflation_gap * 1.5) + (output_gap * 0.5)
        
        # Limit the size of any single adjustment
        rate_change = clamp(rate_change, -0.01, 0.01)
        
        # Apply the adjustment
        new_rate = self.interest_rate + rate_change
        self.set_interest_rate(new_rate)
        
        # Set policy stance
        if rate_change > 0.001:
            self.policy_stance = "Contractionary"
        elif rate_change < -0.001:
            self.policy_stance = "Expansionary"
        else:
            self.policy_stance = "Neutral"
            
    def update_money_supply(self, gdp_growth):
        """Update money supply based on economic growth and policy"""
        # Base money supply growth follows economic growth
        base_growth = gdp_growth
        
        # Adjust based on policy stance
        if self.policy_stance == "Expansionary":
            policy_adjustment = 0.01
        elif self.policy_stance == "Contractionary":
            policy_adjustment = -0.005
        else:
            policy_adjustment = 0
            
        # Calculate growth rate
        growth_rate = base_growth + policy_adjustment
        
        # Apply growth rate
        self.money_supply *= (1 + growth_rate)
        
    def update_monthly(self, economy):
        """Monthly central bank update"""
        # Adjust monetary policy based on economic conditions
        self.adjust_policy(
            economy.global_conditions["inflation"],
            economy.global_conditions["unemployment"],
            economy.global_conditions["economic_growth"]
        )
        
        # Update money supply
        self.update_money_supply(economy.global_conditions["economic_growth"])
        
        # Track history
        self.interest_rate_history.append(self.interest_rate)
        self.inflation_history.append(economy.global_conditions["inflation"])
        self.money_supply_history.append(self.money_supply)
        
    def get_summary_stats(self):
        """Get summary statistics for display"""
        return {
            "interest_rate": f"{self.interest_rate*100:.2f}%",
            "inflation_target": f"{self.inflation_target*100:.1f}%",
            "policy_stance": self.policy_stance,
            "money_supply": self.money_supply
        }

# -----------------------------
# Housing Market Class
# -----------------------------
class HousingMarket:
    def __init__(self):
        self.avg_house_price = 250000
        self.total_housing_units = 0
        self.price_growth_rate = 0.03  # Annual growth rate
        self.monthly_transactions = 0
        self.bubble_factor = 0  # 0-1 scale, where 1 is extreme bubble
        self.rental_yield = 0.04  # Annual rental yield
        
        # Regional variation
        self.regional_prices = {
            "Urban": self.avg_house_price * 1.3,
            "Suburban": self.avg_house_price * 1.0,
            "Rural": self.avg_house_price * 0.7
        }
        
        # Track history
        self.price_history = [self.avg_house_price]
        self.transaction_history = [0]
        self.rental_yield_history = [self.rental_yield]
        self.is_bubble = False
        self.bubble_history = [0]  # Track bubble factor over time
        
    def update_prices(self, economy):
        """Update housing prices based on economic conditions"""
        # Base monthly growth rate (from annual)
        base_growth = self.price_growth_rate / 12
        
        # Adjust for interest rates - lower rates increase prices
        interest_effect = -0.5 * (economy.central_bank.interest_rate - 0.05)
        
        # Economic growth affects prices
        growth_effect = economy.global_conditions["economic_growth"] * 2
        
        # Employment affects demand
        employment_effect = -0.5 * (economy.global_conditions["unemployment"] - 0.05)
        
        # Calculate price adjustment
        price_adjustment = (
            base_growth + 
            interest_effect + 
            growth_effect +
            employment_effect
        )
        
        # Apply bubble dynamics
        bubble_growth = self.update_bubble_factor(economy)
        price_adjustment += bubble_growth
        
        # Apply the price adjustment
        self.avg_house_price *= (1 + price_adjustment)
        
        # Update regional prices
        for region in self.regional_prices:
            regional_adjustment = price_adjustment
            if region == "Urban" and economy.global_conditions["economic_growth"] > 0.01:
                regional_adjustment += 0.002  # Urban areas grow faster in good times
            self.regional_prices[region] *= (1 + regional_adjustment)
            
        # Update rental yields (typically move inversely to prices)
        rental_adjustment = -0.2 * price_adjustment  # Rents are stickier than prices
        self.rental_yield = max(0.02, min(0.08, self.rental_yield * (1 + rental_adjustment)))
        
        # Track history
        self.price_history.append(self.avg_house_price)
        self.rental_yield_history.append(self.rental_yield)
        self.bubble_history.append(self.bubble_factor)
        
    def update_bubble_factor(self, economy):
        """Update housing bubble dynamics"""
        # Start with current bubble factor
        current_bubble = self.bubble_factor
        
        # Factors that increase bubble risk
        bubble_pressure = 0
        
        # Low interest rates can fuel bubbles
        if economy.central_bank.interest_rate < 0.03:
            bubble_pressure += 0.01
            
        # High price-to-income ratios increase bubble risk
        avg_income = getattr(economy, 'avg_income', 30000)  # Default if not set

        if economy.avg_income > 0:  # Add this check to prevent division by zero
            price_to_income = self.avg_house_price / economy.avg_income
            if price_to_income > 8:
                bubble_pressure += 0.01
        else:
            # Default behavior when avg_income is zero
            bubble_pressure += 0.005  # Add moderate pressure
            
        # Rapid price growth increases bubble risk
        if len(self.price_history) > 12:
            annual_growth = self.price_history[-1] / self.price_history[-13] - 1
            if annual_growth > 0.1:  # 10% annual growth
                bubble_pressure += 0.02 * (annual_growth / 0.1)  # More pressure for higher growth
                
        # Excessive mortgage lending increases bubble risk
        if economy.global_conditions.get("mortgage_lending_growth", 0) > 0.08:
            bubble_pressure += 0.01
            
        # Factors that reduce bubble risk
        bubble_resistance = 0
        
        # High interest rates can cool bubbles
        if economy.central_bank.interest_rate > 0.06:
            bubble_resistance += 0.01
            
        # Regulatory measures
        if economy.global_conditions.get("housing_regulation_strength", 0.5) > 0.7:
            bubble_resistance += 0.01
            
        # Recession tends to burst bubbles
        if economy.global_conditions["economic_growth"] < -0.01:
            bubble_resistance += 0.03
            
        # Calculate net bubble pressure
        net_pressure = bubble_pressure - bubble_resistance
        
        # Update bubble factor with inertia and limiting
        new_bubble = current_bubble + net_pressure
        new_bubble = clamp(new_bubble, 0, 1)
        
        # Bubble burst mechanics
        if current_bubble > 0.7 and random.random() < current_bubble * 0.05:
            # Chance of bubble bursting increases as bubble gets larger
            print("Housing bubble burst!")
            new_bubble = max(0, current_bubble - random.uniform(0.5, 0.8))
            
            # Price crash during bubble burst
            bubble_burst_return = -0.15 * current_bubble
            self.is_bubble = False
            return bubble_burst_return
            
        self.bubble_factor = new_bubble
        self.is_bubble = (new_bubble > 0.5)
        
        # Return additional growth due to bubble
        if new_bubble > 0.5:
            return new_bubble * 0.01  # Up to 1% additional monthly growth at max bubble
        return 0
        
    def calculate_mortgage_affordability(self, income, interest_rate, term_years=25):
        """Calculate mortgage affordability at current interest rate"""
        # Monthly interest rate
        monthly_rate = interest_rate / 12
        
        # Number of payments
        num_payments = term_years * 12
        
        # Maximum affordable payment (33% of income)
        max_payment = income * 0.33
        
        # Maximum loan amount
        if monthly_rate > 0:
            max_loan = max_payment * ((1 - (1 + monthly_rate) ** -num_payments) / monthly_rate)
        else:
            max_loan = max_payment * num_payments
            
        return max_loan
    
    def update_monthly(self, economy):
        """Monthly update for housing market"""
        # Update prices
        self.update_prices(economy)
        
        # Simulate transactions
        self.simulate_transactions(economy)
        
        # Calculate impact on broader economy
        self.calculate_economic_impact(economy)
        
    def simulate_transactions(self, economy):
        """Simulate housing market transactions"""
        # Base transaction volume adjusted for economic conditions
        base_volume = economy.population * 0.002  # 0.2% monthly turnover
        
        # Adjust for interest rates
        interest_factor = 1.0
        if economy.central_bank.interest_rate < 0.03:
            interest_factor = 1.2
        elif economy.central_bank.interest_rate > 0.07:
            interest_factor = 0.8
            
        # Adjust for economic growth
        growth_factor = 1.0
        if economy.global_conditions["economic_growth"] > 0.02:
            growth_factor = 1.2
        elif economy.global_conditions["economic_growth"] < 0:
            growth_factor = 0.7
            
        # Adjust for bubble conditions
        bubble_factor = 1.0
        if self.bubble_factor > 0.8:
            bubble_factor = 0.8  # Transactions slow at bubble peak
        elif self.bubble_factor > 0.5:
            bubble_factor = 1.3  # Transactions increase during bubble formation
            
        # Calculate transactions
        self.monthly_transactions = base_volume * interest_factor * growth_factor * bubble_factor
        
        # Track history
        self.transaction_history.append(self.monthly_transactions)
        
    def calculate_economic_impact(self, economy):
        """Calculate housing market impact on broader economy"""
        # Housing wealth effect on consumption
        price_change = 0
        if len(self.price_history) > 1:
            price_change = self.price_history[-1] / self.price_history[-2] - 1
        
        # Housing wealth effect: price change -> consumption change
        wealth_effect = price_change * 0.2
        
        # Apply wealth effect to consumer confidence
        economy.global_conditions["wealth_effect"] = wealth_effect
        
        # Construction activity based on price growth
        construction_stimulus = 0
        if price_change > 0.01:  # If prices rising significantly
            construction_stimulus = price_change * 0.5
            
        economy.global_conditions["construction_stimulus"] = construction_stimulus
        
    def get_summary_stats(self):
        """Get summary statistics for display"""
        return {
            "avg_price": format_currency(self.avg_house_price),
            "annual_growth": f"{((self.price_history[-1] / self.price_history[-13]) - 1) * 100:.1f}%" if len(self.price_history) >= 13 else "N/A",
            "rental_yield": f"{self.rental_yield * 100:.2f}%",
            "monthly_transactions": int(self.monthly_transactions),
            "bubble_risk": f"{self.bubble_factor * 100:.1f}%",
            "is_bubble": "Yes" if self.is_bubble else "No",
            "urban_premium": f"{(self.regional_prices['Urban'] / self.regional_prices['Suburban'] - 1) * 100:.1f}%"
        }

# -----------------------------
# Stock Market Class
# -----------------------------
class StockMarket:
    def __init__(self):
        self.public_companies = []  # List of public companies
        self.market_index = 1000.0  # Starting index value
        self.market_sentiment = 0.6  # 0-1 scale (bearish to bullish)
        self.volatility = 0.05  # Market volatility
        self.trading_volume = 0
        
        # Track history
        self.index_history = [self.market_index]
        self.sentiment_history = [self.market_sentiment]
        self.volatility_history = [self.volatility]
        self.volume_history = [0]
        
    def add_public_company(self, company):
        """Add a company to the public market"""
        if company not in self.public_companies:
            self.public_companies.append(company)
            
    def update_market_index(self):
        """Update the overall market index"""
        if not self.public_companies:
            return
            
        # Calculate market cap weighted index
        total_market_cap = sum(company.value for company in self.public_companies)
        avg_price_change = sum((company.share_price / company.share_price_history[-2] if len(company.share_price_history) > 1 else 1)
                              for company in self.public_companies) / len(self.public_companies)
                              
        self.market_index *= avg_price_change
        self.index_history.append(self.market_index)
        
    def update_market_sentiment(self, economy):
        """Update market sentiment based on economic conditions"""
        # Start with current sentiment
        current_sentiment = self.market_sentiment
        
        # Factors that improve sentiment
        positive_factors = 0
        
        # Economic growth
        if economy.global_conditions["economic_growth"] > 0.01:
            positive_factors += 0.02
            
        # Low unemployment
        if economy.global_conditions["unemployment"] < 0.05:
            positive_factors += 0.01
            
        # Corporate profitability
        avg_profitability = sum(company.profitability for company in self.public_companies) / len(self.public_companies) if self.public_companies else 0
        if avg_profitability > 0.1:
            positive_factors += 0.02
            
        # Factors that reduce sentiment
        negative_factors = 0
        
        # High inflation
        if economy.global_conditions["inflation"] > 0.04:
            negative_factors += 0.02
            
        # Rising interest rates
        if len(economy.central_bank.interest_rate_history) > 1 and economy.central_bank.interest_rate > economy.central_bank.interest_rate_history[-2]:
            negative_factors += 0.01
            
        # High volatility
        if self.volatility > 0.1:
            negative_factors += 0.01
            
        # Update sentiment with inertia
        new_sentiment = current_sentiment * 0.8 + (current_sentiment + positive_factors - negative_factors) * 0.2
        self.market_sentiment = clamp(new_sentiment, 0.1, 0.9)
        
        # Update volatility
        self.update_volatility(economy)
        
        # Track history
        self.sentiment_history.append(self.market_sentiment)
        
    def update_volatility(self, economy):
        """Update market volatility based on economic conditions"""
        # Base volatility adjustment
        volatility_adjustment = 0
        
        # Sentiment extremes increase volatility
        if self.market_sentiment < 0.2 or self.market_sentiment > 0.8:
            volatility_adjustment += 0.01
            
        # Economic uncertainty increases volatility
        if abs(economy.global_conditions["economic_growth"]) > 0.03:
            volatility_adjustment += 0.01
            
        # Inflation increases volatility
        if economy.global_conditions["inflation"] > 0.05:
            volatility_adjustment += 0.01
            
        # Mean reversion in volatility
        reversion = (0.05 - self.volatility) * 0.1
        
        # Update volatility
        self.volatility = max(0.02, self.volatility + volatility_adjustment + reversion)
        
        # Track history
        self.volatility_history.append(self.volatility)
        
    def simulate_trading(self):
        """Simulate trading activity"""
        # Base volume scaled by number of companies
        base_volume = len(self.public_companies) * 100000
        
        # Adjust for sentiment (higher in extreme markets)
        sentiment_factor = 1.0 + abs(self.market_sentiment - 0.5) * 2
        
        # Adjust for volatility
        volatility_factor = 1.0 + self.volatility * 5
        
        # Calculate volume
        self.trading_volume = base_volume * sentiment_factor * volatility_factor
        
        # Track history
        self.volume_history.append(self.trading_volume)
        
    def update_monthly(self, economy):
        """Monthly update for stock market"""
        # Update market sentiment
        self.update_market_sentiment(economy)
        
        # Update market index
        self.update_market_index()
        
        # Simulate trading
        self.simulate_trading()
        
        # Calculate impact on broader economy
        self.calculate_economic_impact(economy)
        
    def calculate_economic_impact(self, economy):
        """Calculate stock market impact on broader economy"""
        # Stock market wealth effect
        market_return = 0
        if len(self.index_history) > 1:
            market_return = self.index_history[-1] / self.index_history[-2] - 1
            
        # Stock wealth effect on consumption (smaller than housing)
        stock_wealth_effect = market_return * 0.1
        
        # Apply wealth effect to economy
        economy.global_conditions["stock_wealth_effect"] = stock_wealth_effect
        
        # Market sentiment affects business investment
        investment_effect = (self.market_sentiment - 0.5) * 0.1
        economy.global_conditions["market_investment_effect"] = investment_effect
        
    def get_summary_stats(self):
        """Get summary statistics for display"""
        return {
            "market_index": round(self.market_index, 2),
            "monthly_return": f"{((self.index_history[-1] / self.index_history[-2]) - 1) * 100:.2f}%" if len(self.index_history) > 1 else "N/A",
            "annual_return": f"{((self.index_history[-1] / self.index_history[-13]) - 1) * 100:.2f}%" if len(self.index_history) > 12 else "N/A",
            "sentiment": f"{self.market_sentiment * 100:.1f}%",
            "volatility": f"{self.volatility * 100:.2f}%",
            "public_companies": len(self.public_companies),
            "trading_volume": format_currency(self.trading_volume, "")
        }

# -----------------------------
# Economy Class
# -----------------------------
class Economy:
    def __init__(self, num_persons=Config.DEFAULT_NUM_PERSONS, num_companies=Config.DEFAULT_NUM_COMPANIES):
        self.num_persons = num_persons
        self.num_companies = num_companies
        self.initial_population = num_persons
        
        # Components
        self.persons = []
        self.companies = []
        self.government = Government()
        self.central_bank = CentralBank()
        self.housing_market = HousingMarket()
        self.stock_market = StockMarket()
        
        # Tracking
        self.current_date = datetime(Config.SIMULATION_START_YEAR, Config.SIMULATION_START_MONTH, 1)
        self.month = 0
        
        # Initial macroeconomic conditions
        self.gdp = 0
        self.gdp_per_capita = 0
        self.avg_income = 0
        self.unemployment_rate = Config.DEFAULT_UNEMPLOYMENT
        self.population = num_persons
        self.government_spending_effect = 0
        self.public_debt_to_gdp = 0
        
        # Sector information
        self.sector_performance = {sector: 1.0 for sector in Config.BUSINESS_SECTORS}
        self.sector_avg_income = {sector: 30000 for sector in Config.BUSINESS_SECTORS}
        
        # Global economic conditions
        self.global_conditions = {
            "inflation": Config.DEFAULT_INFLATION,
            "economic_growth": 0.02,
            "unemployment": Config.DEFAULT_UNEMPLOYMENT,
            "consumer_confidence": 0.6,
            "wealth_effect": 0,
            "construction_stimulus": 0,
            "stock_wealth_effect": 0,
            "market_investment_effect": 0,
            "social_unrest": 0.01,
            "crime": 0.02,
            "mortgage_lending_growth": 0.03,
            "housing_regulation_strength": 0.5,
            "demand_multiplier": 1.0
        }
        
        # Business types dictionary
        self.business_types = {
            'small_retail': {'sector': 'Retail', 'likelihood': 0.435, 'base_revenue': 1000, 'cost_multiplier': 0.84},
            'tech_startup': {'sector': 'Technology', 'likelihood': 0.082, 'base_revenue': 8000, 'cost_multiplier': 1.30},
            'family_farm': {'sector': 'Agriculture', 'likelihood': 0.335, 'base_revenue': 1400, 'cost_multiplier': 0.74},
            'local_entertainment': {'sector': 'Entertainment', 'likelihood': 0.235, 'base_revenue': 2400, 'cost_multiplier': 0.94},
            'tourism_agency': {'sector': 'Entertainment', 'likelihood': 0.175, 'base_revenue': 3400, 'cost_multiplier': 1.04},
            'manufacturing_plant': {'sector': 'Manufacturing', 'likelihood': 0.040, 'base_revenue': 9000, 'cost_multiplier': 1.60},
            'financial_service': {'sector': 'Financial', 'likelihood': 0.024, 'base_revenue': 11800, 'cost_multiplier': 1.90},
            'sports_club': {'sector': 'Entertainment', 'likelihood': 0.090, 'base_revenue': 2700, 'cost_multiplier': 1.14},
            'art_studio': {'sector': 'Entertainment', 'likelihood': 0.140, 'base_revenue': 1950, 'cost_multiplier': 0.84},
            'construction_firm': {'sector': 'Construction', 'likelihood': 0.072, 'base_revenue': 4700, 'cost_multiplier': 1.34},
            'environmental_consultancy': {'sector': 'Technology', 'likelihood': 0.062, 'base_revenue': 4200, 'cost_multiplier': 1.24},
            'security_firm': {'sector': 'Technology', 'likelihood': 0.105, 'base_revenue': 3600, 'cost_multiplier': 1.16},
            'defence_contractor': {'sector': 'Manufacturing', 'likelihood': 0.028, 'base_revenue': 9200, 'cost_multiplier': 1.62},
            'healthcare_provider': {'sector': 'Healthcare', 'likelihood': 0.085, 'base_revenue': 5400, 'cost_multiplier': 1.15},
            'education_service': {'sector': 'Education', 'likelihood': 0.15, 'base_revenue': 4000, 'cost_multiplier': 1.0},
            'food_service': {'sector': 'Food', 'likelihood': 0.22, 'base_revenue': 3200, 'cost_multiplier': 0.9},
            'transport_company': {'sector': 'Transport', 'likelihood': 0.11, 'base_revenue': 4800, 'cost_multiplier': 1.1},
            'energy_provider': {'sector': 'Energy', 'likelihood': 0.04, 'base_revenue': 7500, 'cost_multiplier': 1.4}
        }
        
        # Macro history tracking
        self.macro_history = {
            "GDP": [],
            "GDP Growth": [],
            "GDP Per Capita": [],
            "Unemployment": [],
            "Inflation": [],
            "Interest Rate": [],
            "Consumer Confidence": [],
            "Housing Price": [],
            "Stock Market Index": [],
            "Government Debt": [],
            "Debt to GDP": [],
            "Budget Balance": [],
            "Average Income": [],
            "Median Income": [],
            "Gini Coefficient": [],
            "Population": [],
            "Business Count": [],
            "Corporate Profits": [],
            "Private Investment": [],
            "Consumption": [],
            "Tax Revenue": [],
            "Government Spending": [],
            "Money Supply": [],
            "Trade Balance": []
        }
        self.gini_coefficient = 0.0  
        self.avg_income = 0.0
        self.median_income = 0.0
        
        # Event system
        self.events = []
        self.event_triggers = self.setup_event_triggers()
        
        # Gemini integration
        if GEMINI_AVAILABLE:
            self.gemini_model = None
            self.setup_gemini()
        
    def setup_gemini(self):
        """Setup Gemini API if available"""
        if GEMINI_AVAILABLE:
            try:
                # Check for API key in environment variables
                api_key = os.environ.get('AIzaSyA6UXqG3EjjoJSVb9Q11hewkFwXPztJfTY')
                if api_key:
                    genai.configure(api_key=api_key)
                    self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
                    print("Gemini AI integration activated.")
                else:
                    print("Gemini API key not found. Place your API key in the GEMINI_API_KEY environment variable.")
            except Exception as e:
                print(f"Failed to initialize Gemini API: {e}")
                
    def ask_gemini(self, prompt):
        """Use Gemini to generate insights or event descriptions"""
        if not GEMINI_AVAILABLE or self.gemini_model is None:
            return "Gemini API not available. Using fallback logic instead."
            
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return "Error generating content with Gemini."
            
    def setup_event_triggers(self):
        """Setup economic event triggers"""
        return {
            "housing_bubble": {
                "condition": lambda: self.housing_market.bubble_factor > 0.8,
                "handler": self.trigger_housing_bubble_event
            },
            "recession": {
                "condition": lambda: (
                    self.global_conditions["economic_growth"] < -0.01 and 
                    self.global_conditions["unemployment"] > 0.07
                ),
                "handler": self.trigger_recession_event
            },
            "technology_boom": {
                "condition": lambda: (
                    sum(1 for c in self.companies if c.sector == "Technology" and c.growth_rate > 0.1) > 
                    len([c for c in self.companies if c.sector == "Technology"]) * 0.3
                ),
                "handler": self.trigger_technology_boom_event
            },
            "central_bank_intervention": {
                "condition": lambda: abs(self.global_conditions["inflation"] - self.central_bank.inflation_target) > 0.03,
                "handler": self.trigger_central_bank_event
            },
            "market_crash": {
                "condition": lambda: (
                    self.stock_market.market_sentiment < 0.3 and 
                    len(self.stock_market.index_history) > 2 and
                    self.stock_market.index_history[-1] / self.stock_market.index_history[-2] < 0.92
                ),
                "handler": self.trigger_market_crash_event
            }
        }
            
    def trigger_housing_bubble_event(self):
        """Handle a housing bubble event"""
        if GEMINI_AVAILABLE and self.gemini_model:
            prompt = f"""
            Create a realistic economic events report about a housing bubble in an economy with these conditions:
            - Current house price: {format_currency(self.housing_market.avg_house_price)}
            - Annual price growth: {((self.housing_market.price_history[-1] / self.housing_market.price_history[-13]) - 1) * 100:.1f}% 
            - Interest rate: {self.central_bank.interest_rate*100:.1f}%
            - Unemployment: {self.global_conditions["unemployment"]*100:.1f}%
            - GDP growth: {self.global_conditions["economic_growth"]*100:.1f}%
            
            The report should include:
            1. Description of housing market conditions
            2. Factors contributing to the bubble
            3. Risks to financial stability
            4. Possible policy responses
            
            Keep it under 200 words and focus on economic impacts.
            """
            event_description = self.ask_gemini(prompt)
        else:
            event_description = (
                "HOUSING BUBBLE ALERT: Housing prices have risen to unsustainable levels, with price-to-income "
                f"ratios exceeding historical norms. Current average price: {format_currency(self.housing_market.avg_house_price)}. "
                "Mortgage lending has expanded rapidly and speculative investment is increasing. "
                "The central bank is monitoring the situation closely. Risk of significant correction is high."
            )
            
        self.events.append({
            "date": self.current_date,
            "type": "Housing Bubble",
            "description": event_description,
            "severity": "High"
        })
        
        # Apply economic effects
        self.global_conditions["consumer_confidence"] *= 0.95  # Reduced confidence
        return True
        
    def trigger_recession_event(self):
        """Handle a recession event"""
        if GEMINI_AVAILABLE and self.gemini_model:
            prompt = f"""
            Create a realistic economic events report about a recession in an economy with these conditions:
            - GDP growth: {self.global_conditions["economic_growth"]*100:.1f}%
            - Unemployment: {self.global_conditions["unemployment"]*100:.1f}%
            - Inflation: {self.global_conditions["inflation"]*100:.1f}%
            - Interest rate: {self.central_bank.interest_rate*100:.1f}%
            - Consumer confidence: {self.global_conditions["consumer_confidence"]*100:.1f}%
            
            The report should include:
            1. Description of economic conditions
            2. Sectors most affected
            3. Employment impact
            4. Fiscal and monetary policy responses
            
            Keep it under 200 words and be realistically pessimistic but not catastrophic.
            """
            event_description = self.ask_gemini(prompt)
        else:
            event_description = (
                "ECONOMIC RECESSION: The economy has entered a recession with negative growth for the second consecutive quarter. "
                f"Unemployment has risen to {self.global_conditions['unemployment']*100:.1f}% and consumer confidence is declining. "
                "Retail and manufacturing sectors are particularly affected. "
                "The government is considering fiscal stimulus measures while the central bank is reviewing interest rate policy."
            )
            
        self.events.append({
            "date": self.current_date,
            "type": "Economic Recession",
            "description": event_description,
            "severity": "High"
        })
        
        # Apply economic effects
        for company in self.companies:
            company.revenue *= random.uniform(0.85, 0.95)  # Revenue reduction
        self.global_conditions["demand_multiplier"] *= 0.9  # Reduced demand
        return True
        
    def trigger_technology_boom_event(self):
        """Handle a technology boom event"""
        if GEMINI_AVAILABLE and self.gemini_model:
            prompt = f"""
            Create a realistic economic events report about a technology sector boom in an economy with these conditions:
            - GDP growth: {self.global_conditions["economic_growth"]*100:.1f}%
            - Tech sector growth: significantly above average
            - Unemployment: {self.global_conditions["unemployment"]*100:.1f}%
            - Interest rate: {self.central_bank.interest_rate*100:.1f}%
            - Stock market performance: {((self.stock_market.index_history[-1] / self.stock_market.index_history[-13]) - 1) * 100:.1f}% annually
            
            The report should include:
            1. Description of technology sector growth
            2. Innovation drivers
            3. Investment patterns
            4. Labor market impacts
            
            Keep it under 200 words and focus on realistic economic implications.
            """
            event_description = self.ask_gemini(prompt)
        else:
            event_description = (
                "TECHNOLOGY BOOM: The technology sector is experiencing extraordinary growth, driving innovation across the economy. "
                "Technology companies are reporting stronger-than-expected earnings and rapid expansion. "
                "Venture capital investment has increased significantly. "
                "Demand for skilled technology workers is high, pushing up wages in the sector. "
                "This boom is contributing positively to overall economic growth and productivity."
            )
            
        self.events.append({
            "date": self.current_date,
            "type": "Technology Boom",
            "description": event_description,
            "severity": "Medium"
        })
        
        # Apply economic effects
        for company in self.companies:
            if company.sector == "Technology":
                company.revenue *= random.uniform(1.05, 1.15)  # Revenue boost
                company.innovation_level = min(1.0, company.innovation_level * 1.05)  # Innovation boost
        return True
        
    def trigger_central_bank_event(self):
        """Handle a central bank intervention event"""
        intervention_type = "rate_increase" if self.global_conditions["inflation"] > self.central_bank.inflation_target else "rate_decrease"
        
        if GEMINI_AVAILABLE and self.gemini_model:
            prompt = f"""
            Create a realistic central bank announcement about {'raising' if intervention_type == 'rate_increase' else 'lowering'} interest rates with these economic conditions:
            - Current interest rate: {self.central_bank.interest_rate*100:.2f}%
            - Inflation: {self.global_conditions["inflation"]*100:.1f}%
            - Inflation target: {self.central_bank.inflation_target*100:.1f}%
            - GDP growth: {self.global_conditions["economic_growth"]*100:.1f}%
            - Unemployment: {self.global_conditions["unemployment"]*100:.1f}%
            
            The announcement should include:
            1. The decision to {'raise' if intervention_type == 'rate_increase' else 'lower'} rates
            2. Rationale for the decision
            3. Economic outlook
            4. Forward guidance
            
            Keep it under 200 words and use formal central bank language.
            """
            event_description = self.ask_gemini(prompt)
        else:
            if intervention_type == "rate_increase":
                event_description = (
                    "CENTRAL BANK RAISES RATES: In response to rising inflation, the central bank has increased the base interest rate "
                    f"to {self.central_bank.interest_rate*100:.2f}%. The bank noted concerns about price stability with inflation "
                    f"currently at {self.global_conditions['inflation']*100:.1f}%, above the {self.central_bank.inflation_target*100:.1f}% target. "
                    "The bank remains committed to bringing inflation back to target while supporting sustainable economic growth."
                )
            else:
                event_description = (
                    "CENTRAL BANK CUTS RATES: To stimulate economic growth, the central bank has reduced the base interest rate "
                    f"to {self.central_bank.interest_rate*100:.2f}%. With inflation at {self.global_conditions['inflation']*100:.1f}%, "
                    f"below the {self.central_bank.inflation_target*100:.1f}% target, the bank sees room for monetary stimulus. "
                    "The bank stated it will monitor economic conditions closely and adjust policy as needed."
                )
                
        self.events.append({
            "date": self.current_date,
            "type": "Central Bank Policy Change",
            "description": event_description,
            "severity": "Medium"
        })
        
        # Apply economic effects
        rate_change = 0.005 if intervention_type == "rate_increase" else -0.005
        self.central_bank.set_interest_rate(self.central_bank.interest_rate + rate_change)
        return True
        
    def trigger_market_crash_event(self):
        """Handle a stock market crash event"""
        if GEMINI_AVAILABLE and self.gemini_model:
            prompt = f"""
            Create a realistic economic events report about a stock market crash with these conditions:
            - Stock market index: {self.stock_market.market_index:.1f} (down sharply)
            - Market sentiment: very bearish
            - Economic growth: {self.global_conditions["economic_growth"]*100:.1f}%
            - Interest rate: {self.central_bank.interest_rate*100:.1f}%
            - Corporate profits trend: {'positive' if sum(c.profit for c in self.companies) > 0 else 'negative'}
            
            The report should include:
            1. Description of market conditions and crash
            2. Sectors most affected
            3. Investor sentiment
            4. Potential economic implications
            
            Keep it under 200 words and be realistic but dramatic.
            """
            event_description = self.ask_gemini(prompt)
        else:
            event_description = (
                "STOCK MARKET CRASH: The stock market has experienced a severe correction, with the main index falling sharply. "
                "Investors are fleeing to safe-haven assets amid concerns about economic growth and corporate profitability. "
                "Trading was volatile with record volumes. Financial and technology stocks are among the hardest hit. "
                "Market analysts are concerned about potential spillover effects to the broader economy through reduced wealth and confidence."
            )
            
        self.events.append({
            "date": self.current_date,
            "type": "Stock Market Crash",
            "description": event_description,
            "severity": "High"
        })
        
        # Apply economic effects
        self.stock_market.market_sentiment *= 0.8  # Severely reduced sentiment
        self.stock_market.volatility *= 1.5  # Increased volatility
        self.global_conditions["consumer_confidence"] *= 0.9  # Reduced consumer confidence
        
        # Reduce share prices
        crash_severity = random.uniform(0.7, 0.85)  # 15-30% crash
        for company in self.companies:
            if company.is_public and hasattr(company, 'share_price'):
                company.share_price *= crash_severity
                
        return True
        
    def check_event_triggers(self):
        """Check for event triggers and execute appropriate handlers"""
        for event_name, event_data in self.event_triggers.items():
            if event_data["condition"]():
                # Check if this event was triggered recently (avoid spam)
                recent_events = [e for e in self.events[-10:] if e["type"] == event_name] if self.events else []
                if not recent_events:
                    event_data["handler"]()
                    
    def initialize_people(self):
        """Initialize the population"""
        # Use multiprocessing to create people in chunks
        pool = multiprocessing.Pool()
        chunks = []
        chunk_size = 200000
        for start in range(0, self.num_persons, chunk_size):
            count = min(chunk_size, self.num_persons - start)
            chunks.append(pool.apply_async(create_person_chunk, args=(start+1, count)))
        pool.close()
        pool.join()
        for res in chunks:
            self.persons.extend(res.get())
            
    def initialize_companies(self):
        """Initialize companies"""
        for i in range(self.num_companies):
            # Choose a business type randomly from our business_types dictionary
            btype = random.choice(list(self.business_types.keys()))
            sector = self.business_types[btype]['sector']
            owner_ed = random.choice(["Secondary", "College", "Bachelor's"])
            
            name = self.generate_company_name(btype, sector)
            business = Business(name, sector, i+1, btype, owner_ed)
            
            # Some companies start as public
            if random.random() < 0.05:  # 5% are public initially
                business.is_public = True
                business.launch_ipo(self)
                
            self.companies.append(business)
            
    def generate_company_name(self, business_type, sector):
        """Generate a realistic company name"""
        prefixes = {
            "Technology": ["Tech", "Digital", "Cyber", "Data", "Smart", "Cloud", "Quantum", "Fusion", "Logic", "Nexus"],
            "Manufacturing": ["Industrial", "Precision", "Forge", "Assembly", "Production", "Fabrication", "Dynamic", "Material", "Steel", "Craft"],
            "Retail": ["Shop", "Market", "Store", "Retail", "Emporium", "Goods", "Consumer", "Trade", "Outlet", "Commerce"],
            "Financial": ["Capital", "Finance", "Credit", "Trust", "Asset", "Equity", "Investment", "Bank", "Wealth", "Money"],
            "Healthcare": ["Health", "Care", "Medical", "Wellness", "Life", "Cure", "Remedy", "Healing", "Vital", "Therapy"],
            "Energy": ["Energy", "Power", "Fuel", "Electric", "Solar", "Wind", "Grid", "Thermo", "Charge", "Volt"],
            "Transport": ["Transport", "Shipping", "Logistics", "Cargo", "Fleet", "Motion", "Transit", "Express", "Journey", "Courier"],
            "Agriculture": ["Farm", "Crop", "Harvest", "Field", "Agro", "Growth", "Seed", "Land", "Nature", "Soil"],
            "Entertainment": ["Fun", "Media", "Play", "Leisure", "Entertainment", "Joy", "Stage", "Show", "Thrill", "Delight"],
            "Construction": ["Build", "Construct", "Structure", "Foundation", "Framework", "Property", "Development", "Project", "Site", "Craft"],
            "Education": ["Learn", "Edu", "Study", "Knowledge", "Wisdom", "Scholar", "Academic", "Teach", "Mind", "Train"],
            "Food": ["Food", "Cuisine", "Taste", "Flavor", "Gourmet", "Meal", "Cook", "Culinary", "Dish", "Savor"]
        }
        
        suffixes = ["Corp", "Ltd", "Inc", "Group", "Partners", "Associates", "International", "Solutions", "Systems", "Industries", "Enterprises"]
        
        sector_prefixes = prefixes.get(sector, ["Global", "Prime", "Core", "Alpha", "Mega", "Superior"])
        prefix = random.choice(sector_prefixes)
        suffix = random.choice(suffixes)
        
        # Sometimes add a middle component
        if random.random() < 0.3:
            middle_components = ["Global", "Advanced", "Premier", "United", "Strategic", "Dynamic", "Professional", "American", "European", "National", "Modern"]
            name = f"{prefix} {random.choice(middle_components)} {suffix}"
        else:
            name = f"{prefix} {suffix}"
            
        return name
        
    def update_people_in_chunks(self, chunk_size=10000):
        """Update all persons in a chunk-based approach"""
        # Only update a sample of people for performance
        max_people_to_update = min(100000, len(self.persons))  # Cap at 100,000 people
        people_to_update = random.sample(self.persons, max_people_to_update)
        
        for i in range(0, len(people_to_update), chunk_size):
            chunk = people_to_update[i:i + chunk_size]
            for person in chunk:
                person.update_monthly(self)
                
    def update_sector_performance(self):
        """Update sector performance based on economic conditions"""
        base_adjustment = self.global_conditions["economic_growth"]
        
        # Update each sector with some randomization and sector-specific factors
        for sector in self.sector_performance:
            # Base adjustment
            adjustment = base_adjustment + random.uniform(-0.01, 0.01)
            
            # Sector-specific adjustments
            if sector == "Technology" and self.global_conditions.get("innovation_rate", 0.02) > 0.03:
                adjustment += 0.01
                
            if sector == "Construction" and self.global_conditions.get("construction_stimulus", 0) > 0:
                adjustment += self.global_conditions["construction_stimulus"]
                
            if sector == "Retail" and self.global_conditions["consumer_confidence"] < 0.4:
                adjustment -= 0.01
                
            if sector == "Financial" and self.central_bank.interest_rate > 0.08:
                adjustment += 0.005
                
            if sector == "Healthcare" and self.government.spending_allocation.get("healthcare", 0) > 0.08:
                adjustment += 0.005
                
            # Apply adjustment with smoothing
            current = self.sector_performance[sector]
            self.sector_performance[sector] = current * 0.8 + (current * (1 + adjustment)) * 0.2
            
            # Ensure reasonable bounds
            self.sector_performance[sector] = clamp(self.sector_performance[sector], 0.5, 2.0)
            
    def update_macro_indicators(self):
        """Update macroeconomic indicators"""
        try:
            # Calculate GDP
            production_gdp = sum(company.revenue for company in self.companies) * 12  # Annualized
            consumption_gdp = sum(person.disposable_income * person.spending_rate for person in self.persons) * 12  # Annualized
            government_gdp = self.government.spending * 12  # Annualized
            
            # Weight the components (simplified approach)
            self.gdp = (production_gdp * 0.5 + consumption_gdp * 0.3 + government_gdp * 0.2)
            
            # GDP per capita
            self.population = max(1, len(self.persons))  # Ensure not zero
            self.gdp_per_capita = self.gdp / self.population
            
            # Calculate GDP growth
            if len(self.macro_history["GDP"]) > 0 and self.macro_history["GDP"][-1] > 0:
                previous_gdp = self.macro_history["GDP"][-1]
                gdp_growth = (self.gdp / previous_gdp) - 1
                self.global_conditions["economic_growth"] = gdp_growth
            else:
                self.global_conditions["economic_growth"] = 0.02  # Initial GDP growth
                
            # Calculate unemployment
            employed = sum(1 for person in self.persons if person.employment_status != "Unemployed")
            self.unemployment_rate = 1 - (employed / self.population) if self.population > 0 else 0.05
            self.global_conditions["unemployment"] = self.unemployment_rate
            
            # Calculate average and median income
            incomes = [person.monthly_income * 12 for person in self.persons]  # Annual incomes
            self.avg_income = sum(incomes) / len(incomes) if incomes else 30000  # Default if no data
            sorted_incomes = sorted(incomes)
            self.median_income = sorted_incomes[len(sorted_incomes) // 2] if sorted_incomes else 25000  # Default if no data
            
            # Calculate Gini coefficient (income inequality)
            self.gini_coefficient = calculate_gini(sorted_incomes) if sorted_incomes else 0.4
            
            # Calculate inflation based on demand-pull and cost-push factors
            self.update_inflation()
            
            # Calculate consumer confidence
            self.update_consumer_confidence()
            
            # Update public debt to GDP ratio
            self.public_debt_to_gdp = self.government.debt / self.gdp if self.gdp > 0 else 0.7
        except Exception as e:
            print(f"Error updating macro indicators: {e}")
            # Set reasonable defaults if calculation fails
            if not hasattr(self, 'gdp') or self.gdp <= 0:
                self.gdp = 1000000000  # 1 billion default
            if not hasattr(self, 'avg_income') or self.avg_income <= 0:
                self.avg_income = 30000  # Default average income
            if not hasattr(self, 'unemployment_rate'):
                self.unemployment_rate = 0.05  # Default 5%
            if not hasattr(self, 'global_conditions'):
                self.global_conditions = {
                    "inflation": 0.02,
                    "economic_growth": 0.02,
                    "unemployment": 0.05,
                    "consumer_confidence": 0.6
                }

    def update_inflation(self):
        """Update inflation rate based on economic factors"""
        # Previous inflation (inertia factor)
        previous_inflation = self.global_conditions["inflation"]
        
        # Demand-pull inflation factors
        output_gap = self.unemployment_rate - Config.NATURAL_UNEMPLOYMENT
        demand_pressure = -output_gap * 0.5  # Negative output gap reduces inflation
        
        # Monetary policy effect
        monetary_effect = (self.central_bank.money_supply / self.gdp) - 0.5 if self.gdp > 0 else 0
        monetary_effect *= 0.1  # Scale down the effect
        
        # Interest rate effect
        interest_effect = -0.2 * (self.central_bank.interest_rate - 0.05)  # Higher rates reduce inflation
        
        # Cost-push factors (simplified)
        cost_push = random.uniform(-0.002, 0.002)  # Random external shocks
        
        # Calculate new inflation rate with inertia
        new_inflation = (
            previous_inflation * 0.7 +  # Inflation inertia
            demand_pressure * 0.1 +
            monetary_effect * 0.1 +
            interest_effect * 0.05 +
            cost_push * 0.05
        )
        
        # Ensure reasonable bounds
        self.global_conditions["inflation"] = clamp(new_inflation, 0.005, 0.15)  # Between 0.5% and 15%
        
    def update_consumer_confidence(self):
        """Update consumer confidence index"""
        # Start with previous confidence
        previous_confidence = self.global_conditions.get("consumer_confidence", 0.6)
        
        # Factors affecting confidence
        confidence_factors = {
            "unemployment": -1.0,  # Higher unemployment reduces confidence
            "inflation": -0.5,     # Higher inflation reduces confidence
            "economic_growth": 2.0, # Higher growth increases confidence
            "stock_wealth_effect": 0.3,  # Stock market performance
            "wealth_effect": 0.3,   # Housing market performance
        }
        
        # Calculate adjustment
        adjustment = 0
        for factor, weight in confidence_factors.items():
            if factor in self.global_conditions:
                adjustment += self.global_conditions[factor] * weight
                
        # Apply adjustment with smoothing
        new_confidence = previous_confidence * 0.8 + (previous_confidence + adjustment) * 0.2
        
        # Ensure reasonable bounds
        self.global_conditions["consumer_confidence"] = clamp(new_confidence, 0.2, 0.9)
        
    def update_monthly(self):
        """Simulate one month in the economy"""
        print(f"Simulating month {self.month+1}: {self.current_date.strftime('%B %Y')}")
        
        # Update sector performance
        self.update_sector_performance()
        
        # Update companies
        for company in self.companies:
            company.update_monthly(self)
            
        # Update people (in chunks)
        self.update_people_in_chunks()
        
        # Update housing market
        self.housing_market.update_monthly(self)
        
        # Update stock market
        self.stock_market.update_monthly(self)
        
        # Update central bank
        self.central_bank.update_monthly(self)
        
        # Update government
        self.government.update_monthly(self)
        
        # Update macroeconomic indicators
        self.update_macro_indicators()
        
        # Check event triggers
        self.check_event_triggers()
        
        # Track macro history
        self.track_macro_history()
        
        # Increment month
        self.month += 1
        self.current_date = self.current_date + timedelta(days=30)
        
    def track_macro_history(self):
        """Track macroeconomic history"""
        self.macro_history["GDP"].append(self.gdp)
        self.macro_history["GDP Growth"].append(self.global_conditions["economic_growth"])
        self.macro_history["GDP Per Capita"].append(self.gdp_per_capita)
        self.macro_history["Unemployment"].append(self.unemployment_rate)
        self.macro_history["Inflation"].append(self.global_conditions["inflation"])
        self.macro_history["Interest Rate"].append(self.central_bank.interest_rate)
        self.macro_history["Consumer Confidence"].append(self.global_conditions["consumer_confidence"])
        self.macro_history["Housing Price"].append(self.housing_market.avg_house_price)
        self.macro_history["Stock Market Index"].append(self.stock_market.market_index)
        self.macro_history["Government Debt"].append(self.government.debt)
        self.macro_history["Debt to GDP"].append(self.public_debt_to_gdp)
        self.macro_history["Budget Balance"].append(self.government.budget_balance)
        self.macro_history["Average Income"].append(self.avg_income)
        self.macro_history["Median Income"].append(self.median_income)
        self.macro_history["Gini Coefficient"].append(self.gini_coefficient)
        self.macro_history["Population"].append(self.population)
        self.macro_history["Business Count"].append(len(self.companies))
        self.macro_history["Corporate Profits"].append(sum(company.profit for company in self.companies))
        self.macro_history["Private Investment"].append(sum(getattr(company, "r_and_d_investment", 0) for company in self.companies))
        self.macro_history["Consumption"].append(sum(person.disposable_income * person.spending_rate for person in self.persons))
        self.macro_history["Tax Revenue"].append(self.government.tax_revenue) 
        self.macro_history["Government Spending"].append(self.government.spending)
        self.macro_history["Money Supply"].append(self.central_bank.money_supply)
        self.macro_history["Trade Balance"].append(self.gdp * random.uniform(-0.05, 0.05))  # Simplified
        
    def get_summary_stats(self):
        """Get summary macroeconomic statistics"""
        return {
            "date": self.current_date.strftime("%B %Y"),
            "gdp": format_currency(self.gdp),
            "gdp_growth": f"{self.global_conditions['economic_growth']*100:.2f}%",
            "gdp_per_capita": format_currency(self.gdp_per_capita),
            "unemployment": f"{self.unemployment_rate*100:.2f}%",
            "inflation": f"{self.global_conditions['inflation']*100:.2f}%",
            "interest_rate": f"{self.central_bank.interest_rate*100:.2f}%",
            "consumer_confidence": f"{self.global_conditions['consumer_confidence']*100:.1f}%",
            "housing_price": format_currency(self.housing_market.avg_house_price),
            "stock_market": round(self.stock_market.market_index, 1),
            "government_debt": format_currency(self.government.debt),
            "debt_to_gdp": f"{self.public_debt_to_gdp*100:.1f}%",
            "budget_balance": format_currency(self.government.budget_balance * 12),  # Annualized
            "avg_income": format_currency(self.avg_income),
            "population": f"{self.population:,}",
            "companies": len(self.companies),
            "gini": f"{getattr(self, 'gini_coefficient', 0.0):.3f}"

        }

# -----------------------------
# Multiprocessing helper functions
# -----------------------------
def create_person_chunk(start_id, count):
    """Create a chunk of Person objects (for multiprocessing)"""
    return [Person(i) for i in range(start_id, start_id + count)]

def calculate_gini(sorted_values):
    """Calculate Gini coefficient from sorted values"""
    if not sorted_values or len(sorted_values) < 2:
        return 0
        
    n = len(sorted_values)
    total_sum = sum(sorted_values)
    cumulative_sum = 0
    cumulative_percent = 0
    gini = 0
    
    for i in range(n):
        percent_population = 1/n
        percent_value = sorted_values[i]/total_sum if total_sum > 0 else 0
        cumulative_percent += percent_population
        cumulative_value_percent = (cumulative_sum + sorted_values[i])/total_sum if total_sum > 0 else 0
        gini += percent_population * (cumulative_percent - cumulative_value_percent)
        cumulative_sum += sorted_values[i]
        
    return gini * 2  # Multiply by 2 to get standard Gini value (0-1)

# -----------------------------
# GUI Components
# -----------------------------
class SplashScreen(QSplashScreen):
    def __init__(self):
        """Create a stylish splash screen for initialization"""
        # Create a pixmap filled with the background color
        pixmap = QPixmap(600, 400)
        pixmap.fill(QColor(Config.COLORS["background"]))
        
        super().__init__(pixmap)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        
        # Create the layout
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Add the title
        title_label = QLabel("UltraEconomy Simulator")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"""
            font-size: 28px;
            color: {Config.COLORS["primary"]};
            font-weight: bold;
            margin-bottom: 10px;
        """)
        layout.addWidget(title_label)
        
        # Add subtitle
        subtitle_label = QLabel("Advanced Economic & Geopolitical Modeling")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet(f"""
            font-size: 14px;
            color: {Config.COLORS["text_secondary"]};
            margin-bottom: 30px;
        """)
        layout.addWidget(subtitle_label)
        
        # Add message label
        self.message_label = QLabel("Initializing...")
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.setStyleSheet(f"""
            font-size: 14px;
            color: {Config.COLORS["text_primary"]};
            margin-bottom: 20px;
        """)
        layout.addWidget(self.message_label)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {Config.COLORS["surface_light"]};
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }}
            
            QProgressBar::chunk {{
                background-color: {Config.COLORS["primary"]};
                border-radius: 5px;
            }}
        """)
        layout.addWidget(self.progress_bar)
        
        # Add loading details
        self.details_label = QLabel("Preparing simulation...")
        self.details_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.details_label.setStyleSheet(f"""
            font-size: 12px;
            color: {Config.COLORS["text_secondary"]};
            margin-top: 10px;
        """)
        layout.addWidget(self.details_label)
        
        # Create a widget to hold the layout
        self.widget = QWidget(self)
        self.widget.setLayout(layout)
        self.widget.setGeometry(0, 0, 600, 400)
        
    def update_progress(self, value, message="", details=""):
        """Update the progress bar and messages"""
        self.progress_bar.setValue(value)
        
        if message:
            self.message_label.setText(message)
            
        if details:
            self.details_label.setText(details)
            
        # Process events to update the UI
        QApplication.processEvents()

class ModernPushButton(QPushButton):
    """Custom styled push button"""
    def __init__(self, text, primary=True, parent=None):
        super().__init__(text, parent)
        self.primary = primary
        self.setStyleSheet(self.get_style())
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setMinimumHeight(36)
        
    def get_style(self):
        if self.primary:
            return f"""
                QPushButton {{
                    background-color: {Config.COLORS["primary"]};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {QColor(Config.COLORS["primary"]).lighter(110).name()};
                }}
                QPushButton:pressed {{
                    background-color: {QColor(Config.COLORS["primary"]).darker(110).name()};
                }}
            """
        else:
            return f"""
                QPushButton {{
                    background-color: {Config.COLORS["surface_light"]};
                    color: {Config.COLORS["text_primary"]};
                    border: 1px solid {Config.COLORS["border"]};
                    border-radius: 4px;
                    padding: 8px 16px;
                }}
                QPushButton:hover {{
                    background-color: {QColor(Config.COLORS["surface_light"]).lighter(110).name()};
                }}
                QPushButton:pressed {{
                    background-color: {QColor(Config.COLORS["surface_light"]).darker(110).name()};
                }}
            """

class ModernSlider(QSlider):
    """Custom styled slider"""
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 8px;
                background: {Config.COLORS["surface_light"]};
                margin: 2px 0;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {Config.COLORS["primary"]};
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {Config.COLORS["primary"]};
                height: 8px;
                border-radius: 4px;
            }}
        """)

class ModernComboBox(QComboBox):
    """Custom styled combo box"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QComboBox {{
                background-color: {Config.COLORS["surface_light"]};
                color: {Config.COLORS["text_primary"]};
                border: 1px solid {Config.COLORS["border"]};
                border-radius: 4px;
                padding: 6px;
                min-width: 6em;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 1px;
                border-left-color: {Config.COLORS["border"]};
                border-left-style: solid;
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {Config.COLORS["surface"]};
                color: {Config.COLORS["text_primary"]};
                selection-background-color: {Config.COLORS["primary"]};
                selection-color: white;
            }}
        """)

class StatsCard(QFrame):
    """Card to display a key statistic"""
    def __init__(self, title, value, description="", trend=0, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setStyleSheet(f"""
            StatsCard {{
                background-color: {Config.COLORS["surface_light"]};
                border-radius: 8px;
                padding: 12px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {Config.COLORS['text_secondary']}; font-size: 12px;")
        layout.addWidget(title_label)
        
        # Value with trend indicator
        value_layout = QHBoxLayout()
        value_label = QLabel(str(value))
        value_label.setStyleSheet(f"color: {Config.COLORS['text_primary']}; font-size: 20px; font-weight: bold;")
        value_layout.addWidget(value_label)
        
        # Add trend if available
        if trend != 0:
            trend_color = Config.COLORS["positive"] if trend > 0 else Config.COLORS["negative"]
            trend_label = QLabel(f"{'+' if trend > 0 else ''}{trend:.1f}%")
            trend_label.setStyleSheet(f"color: {trend_color}; font-size: 12px; padding-left: 4px;")
            value_layout.addWidget(trend_label)
            value_layout.addStretch()
            
        layout.addLayout(value_layout)
        
        # Description
        if description:
            desc_label = QLabel(description)
            desc_label.setStyleSheet(f"color: {Config.COLORS['text_secondary']}; font-size: 11px;")
            layout.addWidget(desc_label)
            
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

class SimulationThread(QThread):
    """Thread for running simulations in the background"""
    update_progress = pyqtSignal(int, str, str)  # value, message, details
    simulation_complete = pyqtSignal()
    
    def __init__(self, economy):
        super().__init__()
        self.economy = economy
        self.running = True
        
    def run(self):
        """Run a single month of simulation"""
        self.economy.update_monthly()
        self.simulation_complete.emit()
        
    def stop(self):
        """Stop the simulation thread"""
        self.running = False
        self.wait()

class MacroStatBarChart(FigureCanvas):
    """Bar chart for macroeconomic statistics"""
    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=Config.COLORS["background"])
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(Config.COLORS["background"])
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
        
        # Set up figure appearance
        self.ax.tick_params(axis='x', colors=Config.COLORS["text_secondary"])
        self.ax.tick_params(axis='y', colors=Config.COLORS["text_secondary"])
        self.ax.spines['bottom'].set_color(Config.COLORS["border"])
        self.ax.spines['top'].set_color(Config.COLORS["border"])
        self.ax.spines['left'].set_color(Config.COLORS["border"])
        self.ax.spines['right'].set_color(Config.COLORS["border"])
        
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()
        
    def plot_bars(self, categories, values, title="", xlabel="", ylabel="", color=Config.COLORS["primary"]):
        """Plot a bar chart with the provided data"""
        self.ax.clear()
        
        bars = self.ax.bar(categories, values, color=color, width=0.6)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            self.ax.annotate(f'{height:.1f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom',
                             color=Config.COLORS["text_secondary"])
        
        # Set titles and labels
        self.ax.set_title(title, color=Config.COLORS["text_primary"], fontsize=12)
        self.ax.set_xlabel(xlabel, color=Config.COLORS["text_secondary"])
        self.ax.set_ylabel(ylabel, color=Config.COLORS["text_secondary"])
        
        # Set background and grid
        self.ax.set_facecolor(Config.COLORS["background"])
        self.ax.grid(axis='y', linestyle='--', alpha=0.3, color=Config.COLORS["chart_grid"])
        
        # Rotate x-axis labels if there are many categories
        if len(categories) > 5:
            plt.xticks(rotation=45, ha='right')
            
        self.draw()

class MacroStatLineChart(FigureCanvas):
    """Line chart for time series macroeconomic statistics"""
    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=Config.COLORS["background"])
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(Config.COLORS["background"])
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
        
        # Set up figure appearance
        self.ax.tick_params(axis='x', colors=Config.COLORS["text_secondary"])
        self.ax.tick_params(axis='y', colors=Config.COLORS["text_secondary"])
        self.ax.spines['bottom'].set_color(Config.COLORS["border"])
        self.ax.spines['top'].set_color(Config.COLORS["border"])
        self.ax.spines['left'].set_color(Config.COLORS["border"])
        self.ax.spines['right'].set_color(Config.COLORS["border"])
        
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()
        
    def plot_line(self, x_values, y_values, title="", xlabel="", ylabel="", color=Config.COLORS["chart_line"]):
        """Plot a line chart with the provided data"""
        self.ax.clear()
        
        # Plot line with gradient and markers
        self.ax.plot(x_values, y_values, marker='o', color=color, linewidth=2, markersize=4)
        
        # Set titles and labels
        self.ax.set_title(title, color=Config.COLORS["text_primary"], fontsize=12)
        self.ax.set_xlabel(xlabel, color=Config.COLORS["text_secondary"])
        self.ax.set_ylabel(ylabel, color=Config.COLORS["text_secondary"])
        
        # Set background and grid
        self.ax.set_facecolor(Config.COLORS["background"])
        self.ax.grid(True, linestyle='--', alpha=0.3, color=Config.COLORS["chart_grid"])
            
        self.draw()
        
    def plot_multi_line(self, x_values, y_values_dict, title="", xlabel="", ylabel=""):
        """Plot multiple lines on the same chart"""
        self.ax.clear()
        
        # Plot each line
        for i, (label, y_values) in enumerate(y_values_dict.items()):
            color = Config.CHART_COLORS[i % len(Config.CHART_COLORS)]
            self.ax.plot(x_values, y_values, marker='o', color=color, linewidth=2, markersize=4, label=label)
        
        # Add legend
        self.ax.legend(facecolor=Config.COLORS["surface_light"], edgecolor=Config.COLORS["border"], 
                      labelcolor=Config.COLORS["text_primary"])
        
        # Set titles and labels
        self.ax.set_title(title, color=Config.COLORS["text_primary"], fontsize=12)
        self.ax.set_xlabel(xlabel, color=Config.COLORS["text_secondary"])
        self.ax.set_ylabel(ylabel, color=Config.COLORS["text_secondary"])
        
        # Set background and grid
        self.ax.set_facecolor(Config.COLORS["background"])
        self.ax.grid(True, linestyle='--', alpha=0.3, color=Config.COLORS["chart_grid"])
            
        self.draw()

class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self, economy):
        super().__init__()
        self.economy = economy
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface"""
        self.setWindowTitle("UltraEconomy - Advanced Economic Simulator")
        self.setGeometry(50, 50, 1600, 900)
        self.setup_style()
        
        # Create central widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Add toolbar at the top
        self.create_toolbar(main_layout)
        
        # Add tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {Config.COLORS["border"]};
                background-color: {Config.COLORS["surface"]};
                border-radius: 5px;
            }}
            QTabBar::tab {{
                background-color: {Config.COLORS["surface_light"]};
                color: {Config.COLORS["text_secondary"]};
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: {Config.COLORS["primary"]};
                color: white;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {QColor(Config.COLORS["surface_light"]).lighter(120).name()};
            }}
        """)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_people_tab()
        self.create_companies_tab()
        self.create_government_tab()
        self.create_central_bank_tab()
        self.create_housing_market_tab()
        self.create_stock_market_tab()
        self.create_statistics_tab()
        self.create_events_tab()
        
        main_layout.addWidget(self.tabs)
        
        # Add status bar
        self.statusBar = QStatusBar()
        self.statusBar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {Config.COLORS["surface"]};
                color: {Config.COLORS["text_secondary"]};
                border-top: 1px solid {Config.COLORS["border"]};
            }}
        """)
        self.setStatusBar(self.statusBar)
        self.update_status_bar()
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        # Create timer for periodic updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.periodic_update)
        self.update_timer.start(Config.UI_REFRESH_RATE)
        
        # Create simulation thread
        self.simulation_thread = None
        
    def setup_style(self):
        """Set up application style"""
        QApplication.setStyle("Fusion")
        
        # Create a dark palette
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(Config.COLORS["background"]))
        dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(Config.COLORS["text_primary"]))
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(Config.COLORS["surface"]))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(Config.COLORS["surface_light"]))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(Config.COLORS["text_primary"]))
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(Config.COLORS["text_secondary"]))
        dark_palette.setColor(QPalette.ColorRole.Text, QColor(Config.COLORS["text_primary"]))
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(Config.COLORS["surface_light"]))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(Config.COLORS["text_primary"]))
        dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(Config.COLORS["text_primary"]))
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(Config.COLORS["primary"]))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(Config.COLORS["primary"]))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(Config.COLORS["text_primary"]))
        
        QApplication.setPalette(dark_palette)

        
        # Set application stylesheet
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {Config.COLORS["background"]};
            }}
            QWidget {{
                background-color: {Config.COLORS["background"]};
                color: {Config.COLORS["text_primary"]};
            }}
            QLabel {{
                background-color: transparent;
                color: {Config.COLORS["text_primary"]};
            }}
            QGroupBox {{
                border: 1px solid {Config.COLORS["border"]};
                border-radius: 4px;
                margin-top: 1.5em;
                padding-top: 0.5em;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: {Config.COLORS["text_primary"]};
            }}
            QScrollArea {{
                border: none;
            }}
            QTableView {{
                background-color: {Config.COLORS["surface"]};
                color: {Config.COLORS["text_primary"]};
                gridline-color: {Config.COLORS["border"]};
                border: 1px solid {Config.COLORS["border"]};
                border-radius: 4px;
            }}
            QTableView::item:selected {{
                background-color: {Config.COLORS["primary"]};
                color: white;
            }}
            QHeaderView {{
                background-color: {Config.COLORS["surface_light"]};
                color: {Config.COLORS["text_primary"]};
            }}
            QHeaderView::section {{
                background-color: {Config.COLORS["surface_light"]};
                padding: 4px;
                border: 1px solid {Config.COLORS["border"]};
                color: {Config.COLORS["text_primary"]};
            }}
        """)
        
    def create_toolbar(self, layout):
        """Create application toolbar"""
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(10, 5, 10, 5)
        
        # Logo/Title
        logo_label = QLabel("🏢 UltraEconomy")
        logo_label.setStyleSheet(f"""
            font-size: 18px;
            font-weight: bold;
            color: {Config.COLORS["primary"]};
            margin-right: 20px;
        """)
        toolbar_layout.addWidget(logo_label)
        
        # Date display
        self.date_label = QLabel("")
        self.date_label.setStyleSheet(f"""
            font-size: 14px;
            color: {Config.COLORS["text_primary"]};
            margin-right: 20px;
        """)
        toolbar_layout.addWidget(self.date_label)
        
        # Key indicators
        self.gdp_label = QLabel("")
        self.gdp_label.setStyleSheet(f"color: {Config.COLORS['text_secondary']};")
        toolbar_layout.addWidget(self.gdp_label)
        
        toolbar_layout.addSpacing(15)
        
        self.unemployment_label = QLabel("")
        self.unemployment_label.setStyleSheet(f"color: {Config.COLORS['text_secondary']};")
        toolbar_layout.addWidget(self.unemployment_label)
        
        toolbar_layout.addSpacing(15)
        
        self.inflation_label = QLabel("")
        self.inflation_label.setStyleSheet(f"color: {Config.COLORS['text_secondary']};")
        toolbar_layout.addWidget(self.inflation_label)
        
        toolbar_layout.addSpacing(15)
        
        self.interest_rate_label = QLabel("")
        self.interest_rate_label.setStyleSheet(f"color: {Config.COLORS['text_secondary']};")
        toolbar_layout.addWidget(self.interest_rate_label)
        
        # Add spacer
        toolbar_layout.addStretch()
        
        # Simulation controls
        self.run_button = ModernPushButton("Run Month", True)
        self.run_button.clicked.connect(self.run_simulation)
        toolbar_layout.addWidget(self.run_button)
        
        speed_label = QLabel("Speed:")
        speed_label.setStyleSheet(f"color: {Config.COLORS['text_secondary']};")
        toolbar_layout.addWidget(speed_label)
        
        self.speed_combo = ModernComboBox()
        self.speed_combo.addItems(["1x", "2x", "5x", "10x"])
        toolbar_layout.addWidget(self.speed_combo)
        
        layout.addWidget(toolbar_widget)
        
    def create_dashboard_tab(self):
        """Create the dashboard tab"""
        dashboard_tab = QWidget()
        layout = QVBoxLayout(dashboard_tab)
        
        # Top row with key statistics cards
        stats_layout = QHBoxLayout()
        
        # We'll update these later with real data
        self.gdp_card = StatsCard("GDP", "£0", "Gross Domestic Product")
        self.gdp_growth_card = StatsCard("GDP Growth", "0.0%", "Quarterly Change", 0)
        self.unemployment_card = StatsCard("Unemployment", "0.0%", "Rate")
        self.inflation_card = StatsCard("Inflation", "0.0%", "Annual Rate")
        self.housing_card = StatsCard("Avg. House Price", "£0", "Housing Market")
        self.stock_card = StatsCard("Stock Market", "0", "Market Index")
        
        stats_layout.addWidget(self.gdp_card)
        stats_layout.addWidget(self.gdp_growth_card)
        stats_layout.addWidget(self.unemployment_card)
        stats_layout.addWidget(self.inflation_card)
        stats_layout.addWidget(self.housing_card)
        stats_layout.addWidget(self.stock_card)
        
        layout.addLayout(stats_layout)
        
        # Middle section with charts
        charts_layout = QHBoxLayout()
        
        # Left chart group - GDP and components
        gdp_group = QGroupBox("GDP Components")
        gdp_layout = QVBoxLayout(gdp_group)
        self.gdp_chart = MacroStatBarChart(width=6, height=4)
        gdp_layout.addWidget(self.gdp_chart)
        charts_layout.addWidget(gdp_group)
        
        # Right chart group - Key trends
        trends_group = QGroupBox("Key Economic Trends")
        trends_layout = QVBoxLayout(trends_group)
        self.trends_chart = MacroStatLineChart(width=6, height=4)
        trends_layout.addWidget(self.trends_chart)
        charts_layout.addWidget(trends_group)
        
        layout.addLayout(charts_layout)
        
        # Bottom section with recent events and notifications
        events_group = QGroupBox("Recent Events")
        events_layout = QVBoxLayout(events_group)
        self.events_list = QListWidget()
        self.events_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {Config.COLORS["surface"]};
                border-radius: 4px;
                padding: 5px;
                border: 1px solid {Config.COLORS["border"]};
            }}
            QListWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {Config.COLORS["border"]};
            }}
            QListWidget::item:selected {{
                background-color: {Config.COLORS["primary"]};
                color: white;
            }}
        """)
        events_layout.addWidget(self.events_list)
        layout.addWidget(events_group)
        
        self.tabs.addTab(dashboard_tab, "Dashboard")
        
    def create_people_tab(self):
        """Create the people/population tab"""
        people_tab = QWidget()
        layout = QVBoxLayout(people_tab)
        
        # Top toolbar with search and filters
        toolbar_layout = QHBoxLayout()
        
        search_label = QLabel("Search:")
        toolbar_layout.addWidget(search_label)
        
        self.people_search = QLineEdit()
        self.people_search.setPlaceholderText("Search by name or ID...")
        self.people_search.setStyleSheet(f"""
            QLineEdit {{
                background-color: {Config.COLORS["surface_light"]};
                color: {Config.COLORS["text_primary"]};
                border: 1px solid {Config.COLORS["border"]};
                border-radius: 4px;
                padding: 5px;
            }}
        """)
        toolbar_layout.addWidget(self.people_search)
        
        filter_label = QLabel("Filter:")
        toolbar_layout.addWidget(filter_label)
        
        self.people_filter = ModernComboBox()
        self.people_filter.addItems(["All", "Employed", "Unemployed", "Business Owners", "High Net Worth"])
        toolbar_layout.addWidget(self.people_filter)
        
        sort_label = QLabel("Sort by:")
        toolbar_layout.addWidget(sort_label)
        
        self.people_sort = ModernComboBox()
        self.people_sort.addItems(["Income (High to Low)", "Income (Low to High)", 
                                  "Net Worth (High to Low)", "Net Worth (Low to High)",
                                  "Age (High to Low)", "Age (Low to High)"])
        toolbar_layout.addWidget(self.people_sort)
        
        toolbar_layout.addStretch()
        
        page_label = QLabel("Page:")
        toolbar_layout.addWidget(page_label)
        
        self.people_page = QSpinBox()
        self.people_page.setMinimum(1)
        self.people_page.setMaximum(1000)  # We'll update this based on filtered data
        self.people_page.setValue(1)
        self.people_page.setStyleSheet(f"""
            QSpinBox {{
                background-color: {Config.COLORS["surface_light"]};
                color: {Config.COLORS["text_primary"]};
                border: 1px solid {Config.COLORS["border"]};
                border-radius: 4px;
                padding: 5px;
            }}
        """)
        toolbar_layout.addWidget(self.people_page)
        
        layout.addLayout(toolbar_layout)
        
        # Table for displaying people
        self.people_table = QTableWidget()
        self.people_table.setColumnCount(10)
        self.people_table.setHorizontalHeaderLabels([
            "ID", "Name", "Age", "Education", "Income", "Net Worth", 
            "Employment", "Job", "Housing", "Happiness"
        ])
        self.people_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.people_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # Name column stretches
        layout.addWidget(self.people_table)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        refresh_button = ModernPushButton("Refresh", False)
        refresh_button.clicked.connect(self.update_people_table)
        buttons_layout.addWidget(refresh_button)
        
        detail_button = ModernPushButton("View Details", True)
        detail_button.clicked.connect(self.show_person_details)
        buttons_layout.addWidget(detail_button)
        
        buttons_layout.addStretch()
        
        stats_button = ModernPushButton("Population Statistics", False)
        stats_button.clicked.connect(self.show_population_stats)
        buttons_layout.addWidget(stats_button)
        
        layout.addLayout(buttons_layout)
        
        self.tabs.addTab(people_tab, "Population")
        
    def create_companies_tab(self):
        """Create the companies/businesses tab"""
        companies_tab = QWidget()
        layout = QVBoxLayout(companies_tab)
        
        # Top toolbar with search and filters
        toolbar_layout = QHBoxLayout()
        
        search_label = QLabel("Search:")
        toolbar_layout.addWidget(search_label)
        
        self.companies_search = QLineEdit()
        self.companies_search.setPlaceholderText("Search by name or ID...")
        self.companies_search.setStyleSheet(f"""
            QLineEdit {{
                background-color: {Config.COLORS["surface_light"]};
                color: {Config.COLORS["text_primary"]};
                border: 1px solid {Config.COLORS["border"]};
                border-radius: 4px;
                padding: 5px;
            }}
        """)
        toolbar_layout.addWidget(self.companies_search)
        
        filter_label = QLabel("Sector:")
        toolbar_layout.addWidget(filter_label)
        
        self.companies_sector_filter = ModernComboBox()
        self.companies_sector_filter.addItems(["All"] + Config.BUSINESS_SECTORS)
        toolbar_layout.addWidget(self.companies_sector_filter)
        
        status_label = QLabel("Status:")
        toolbar_layout.addWidget(status_label)
        
        self.companies_status_filter = ModernComboBox()
        self.companies_status_filter.addItems(["All", "Public", "Private", "Profitable", "Loss-Making"])
        toolbar_layout.addWidget(self.companies_status_filter)
        
        sort_label = QLabel("Sort by:")
        toolbar_layout.addWidget(sort_label)
        
        self.companies_sort = ModernComboBox()
        self.companies_sort.addItems(["Revenue (High to Low)", "Profit (High to Low)", 
                                     "Employees (High to Low)", "Value (High to Low)",
                                     "Growth (High to Low)"])
        toolbar_layout.addWidget(self.companies_sort)
        
        toolbar_layout.addStretch()
        
        page_label = QLabel("Page:")
        toolbar_layout.addWidget(page_label)
        
        self.companies_page = QSpinBox()
        self.companies_page.setMinimum(1)
        self.companies_page.setMaximum(100)  # We'll update this based on filtered data
        self.companies_page.setValue(1)
        self.companies_page.setStyleSheet(f"""
            QSpinBox {{
                background-color: {Config.COLORS["surface_light"]};
                color: {Config.COLORS["text_primary"]};
                border: 1px solid {Config.COLORS["border"]};
                border-radius: 4px;
                padding: 5px;
            }}
        """)
        toolbar_layout.addWidget(self.companies_page)
        
        layout.addLayout(toolbar_layout)
        
        # Table for displaying companies
        self.companies_table = QTableWidget()
        self.companies_table.setColumnCount(10)
        self.companies_table.setHorizontalHeaderLabels([
            "ID", "Name", "Sector", "Revenue", "Profit", "Margin", 
            "Employees", "Value", "Public", "Growth"
        ])
        self.companies_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.companies_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # Name column stretches
        layout.addWidget(self.companies_table)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        refresh_button = ModernPushButton("Refresh", False)
        refresh_button.clicked.connect(self.update_companies_table)
        buttons_layout.addWidget(refresh_button)
        
        detail_button = ModernPushButton("View Details", True)
        detail_button.clicked.connect(self.show_company_details)
        buttons_layout.addWidget(detail_button)
        
        buttons_layout.addStretch()
        
        stats_button = ModernPushButton("Industry Statistics", False)
        stats_button.clicked.connect(self.show_industry_stats)
        buttons_layout.addWidget(stats_button)
        
        layout.addLayout(buttons_layout)
        
        self.tabs.addTab(companies_tab, "Companies")
        
    def create_government_tab(self):
        """Create the government/fiscal policy tab"""
        govt_tab = QWidget()
        layout = QVBoxLayout(govt_tab)
        
        # Split into two columns
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Fiscal tools and policy controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Tax system
        tax_group = QGroupBox("Tax System")
        tax_layout = QVBoxLayout(tax_group)
        
        # Income tax brackets
        income_tax_header = QLabel("Income Tax Brackets")
        income_tax_header.setStyleSheet(f"font-weight: bold; color: {Config.COLORS['primary']};")
        tax_layout.addWidget(income_tax_header)
        
        self.income_tax_widgets = []
        
        for i, bracket in enumerate(self.economy.government.income_tax_brackets):
            bracket_widget = QWidget()
            bracket_layout = QHBoxLayout(bracket_widget)
            bracket_layout.setContentsMargins(0, 0, 0, 0)
            
            if i == 0:
                bracket_label = QLabel(f"£0 - £{bracket[1]:,}")
            elif i == len(self.economy.government.income_tax_brackets) - 1:
                bracket_label = QLabel(f"Over £{bracket[0]:,}")
            else:
                bracket_label = QLabel(f"£{bracket[0]:,} - £{bracket[1]:,}")
                
            bracket_layout.addWidget(bracket_label, 2)
            
            rate_slider = ModernSlider(Qt.Orientation.Horizontal)
            rate_slider.setMinimum(0)
            rate_slider.setMaximum(100)
            rate_slider.setValue(int(bracket[2] * 100))
            bracket_layout.addWidget(rate_slider, 3)
            
            rate_label = QLabel(f"{bracket[2]*100:.1f}%")
            bracket_layout.addWidget(rate_label, 1)
            
            # Update label when slider changes
            rate_slider.valueChanged.connect(lambda val, label=rate_label: label.setText(f"{val:.1f}%"))
            
            tax_layout.addWidget(bracket_widget)
            self.income_tax_widgets.append((bracket_label, rate_slider, rate_label))
            
        # Corporate tax rate
        corp_tax_widget = QWidget()
        corp_tax_layout = QHBoxLayout(corp_tax_widget)
        corp_tax_layout.setContentsMargins(0, 0, 0, 0)
        
        corp_tax_label = QLabel("Corporate Tax Rate")
        corp_tax_layout.addWidget(corp_tax_label, 2)
        
        self.corp_tax_slider = ModernSlider(Qt.Orientation.Horizontal)
        self.corp_tax_slider.setMinimum(0)
        self.corp_tax_slider.setMaximum(50)
        self.corp_tax_slider.setValue(int(self.economy.government.corporation_tax_rate * 100))
        corp_tax_layout.addWidget(self.corp_tax_slider, 3)
        
        self.corp_tax_value = QLabel(f"{self.economy.government.corporation_tax_rate*100:.1f}%")
        corp_tax_layout.addWidget(self.corp_tax_value, 1)
        
        self.corp_tax_slider.valueChanged.connect(lambda val: self.corp_tax_value.setText(f"{val:.1f}%"))
        
        tax_layout.addWidget(corp_tax_widget)
        
        # Sales tax rate
        sales_tax_widget = QWidget()
        sales_tax_layout = QHBoxLayout(sales_tax_widget)
        sales_tax_layout.setContentsMargins(0, 0, 0, 0)
        
        sales_tax_label = QLabel("Sales Tax / VAT Rate")
        sales_tax_layout.addWidget(sales_tax_label, 2)
        
        self.sales_tax_slider = ModernSlider(Qt.Orientation.Horizontal)
        self.sales_tax_slider.setMinimum(0)
        self.sales_tax_slider.setMaximum(40)
        self.sales_tax_slider.setValue(int(self.economy.government.sales_tax_rate * 100))
        sales_tax_layout.addWidget(self.sales_tax_slider, 3)
        
        self.sales_tax_value = QLabel(f"{self.economy.government.sales_tax_rate*100:.1f}%")
        sales_tax_layout.addWidget(self.sales_tax_value, 1)
        
        self.sales_tax_slider.valueChanged.connect(lambda val: self.sales_tax_value.setText(f"{val:.1f}%"))
        
        tax_layout.addWidget(sales_tax_widget)
        
        # National insurance rate
        ni_widget = QWidget()
        ni_layout = QHBoxLayout(ni_widget)
        ni_layout.setContentsMargins(0, 0, 0, 0)
        
        ni_label = QLabel("National Insurance Rate")
        ni_layout.addWidget(ni_label, 2)
        
        self.ni_slider = ModernSlider(Qt.Orientation.Horizontal)
        self.ni_slider.setMinimum(0)
        self.ni_slider.setMaximum(30)
        self.ni_slider.setValue(int(self.economy.government.national_insurance_rate * 100))
        ni_layout.addWidget(self.ni_slider, 3)
        
        self.ni_value = QLabel(f"{self.economy.government.national_insurance_rate*100:.1f}%")
        ni_layout.addWidget(self.ni_value, 1)
        
        self.ni_slider.valueChanged.connect(lambda val: self.ni_value.setText(f"{val:.1f}%"))
        
        tax_layout.addWidget(ni_widget)
        
        # Apply button for tax changes
        apply_tax_button = ModernPushButton("Apply Tax Changes", True)
        apply_tax_button.clicked.connect(self.apply_tax_changes)
        tax_layout.addWidget(apply_tax_button)
        
        left_layout.addWidget(tax_group)
        
        # Government spending
        spending_group = QGroupBox("Government Spending (% of GDP)")
        spending_layout = QVBoxLayout(spending_group)
        
        self.spending_widgets = {}
        
        for dept, percentage in self.economy.government.spending_allocation.items():
            dept_widget = QWidget()
            dept_layout = QHBoxLayout(dept_widget)
            dept_layout.setContentsMargins(0, 0, 0, 0)
            
            dept_label = QLabel(dept.capitalize())
            dept_layout.addWidget(dept_label, 2)
            
            dept_slider = ModernSlider(Qt.Orientation.Horizontal)
            dept_slider.setMinimum(0)
            dept_slider.setMaximum(200)  # Up to 20% of GDP
            dept_slider.setValue(int(percentage * 1000))  # Multiply by 1000 for precision
            dept_layout.addWidget(dept_slider, 3)
            
            dept_value = QLabel(f"{percentage*100:.1f}%")
            dept_layout.addWidget(dept_value, 1)
            
            # Update label when slider changes
            dept_slider.valueChanged.connect(lambda val, label=dept_value: label.setText(f"{val/10:.1f}%"))
            
            spending_layout.addWidget(dept_widget)
            self.spending_widgets[dept] = (dept_slider, dept_value)
            
        # Apply button for spending changes
        apply_spending_button = ModernPushButton("Apply Spending Changes", True)
        apply_spending_button.clicked.connect(self.apply_spending_changes)
        spending_layout.addWidget(apply_spending_button)
        
        left_layout.addWidget(spending_group)
        
        # Add spacer
        left_layout.addStretch()
        
        # Right side - Current status and statistics
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Current fiscal status
        fiscal_group = QGroupBox("Fiscal Status")
        fiscal_layout = QVBoxLayout(fiscal_group)
        
        # Total revenue
        revenue_widget = QWidget()
        revenue_layout = QHBoxLayout(revenue_widget)
        revenue_layout.setContentsMargins(0, 0, 0, 0)
        
        revenue_label = QLabel("Total Tax Revenue:")
        revenue_label.setStyleSheet("font-weight: bold;")
        revenue_layout.addWidget(revenue_label)
        
        self.revenue_value = QLabel("£0")
        revenue_layout.addWidget(self.revenue_value)
        revenue_layout.addStretch()
        
        fiscal_layout.addWidget(revenue_widget)
        
        # Total spending
        spending_widget = QWidget()
        spending_layout = QHBoxLayout(spending_widget)
        spending_layout.setContentsMargins(0, 0, 0, 0)
        
        spending_label = QLabel("Total Government Spending:")
        spending_label.setStyleSheet("font-weight: bold;")
        spending_layout.addWidget(spending_label)
        
        self.spending_value = QLabel("£0")
        spending_layout.addWidget(self.spending_value)
        spending_layout.addStretch()
        
        fiscal_layout.addWidget(spending_widget)
        
        # Budget balance
        balance_widget = QWidget()
        balance_layout = QHBoxLayout(balance_widget)
        balance_layout.setContentsMargins(0, 0, 0, 0)
        
        balance_label = QLabel("Budget Balance:")
        balance_label.setStyleSheet("font-weight: bold;")
        balance_layout.addWidget(balance_label)
        
        self.balance_value = QLabel("£0")
        balance_layout.addWidget(self.balance_value)
        balance_layout.addStretch()
        
        fiscal_layout.addWidget(balance_widget)
        
        # National debt
        debt_widget = QWidget()
        debt_layout = QHBoxLayout(debt_widget)
        debt_layout.setContentsMargins(0, 0, 0, 0)
        
        debt_label = QLabel("National Debt:")
        debt_label.setStyleSheet("font-weight: bold;")
        debt_layout.addWidget(debt_label)
        
        self.debt_value = QLabel("£0")
        debt_layout.addWidget(self.debt_value)
        debt_layout.addStretch()
        
        fiscal_layout.addWidget(debt_widget)
        
        # Debt to GDP ratio
        debt_gdp_widget = QWidget()
        debt_gdp_layout = QHBoxLayout(debt_gdp_widget)
        debt_gdp_layout.setContentsMargins(0, 0, 0, 0)
        
        debt_gdp_label = QLabel("Debt to GDP Ratio:")
        debt_gdp_label.setStyleSheet("font-weight: bold;")
        debt_gdp_layout.addWidget(debt_gdp_label)
        
        self.debt_gdp_value = QLabel("0%")
        debt_gdp_layout.addWidget(self.debt_gdp_value)
        debt_gdp_layout.addStretch()
        
        fiscal_layout.addWidget(debt_gdp_widget)
        
        right_layout.addWidget(fiscal_group)
        
        # Charts for tax revenue and spending
        charts_group = QGroupBox("Fiscal Charts")
        charts_layout = QVBoxLayout(charts_group)
        
        # Tax revenue breakdown
        tax_chart_label = QLabel("Tax Revenue Breakdown")
        tax_chart_label.setStyleSheet("font-weight: bold;")
        charts_layout.addWidget(tax_chart_label)
        
        self.tax_chart = MacroStatBarChart(width=5, height=3)
        charts_layout.addWidget(self.tax_chart)
        
        # Government spending breakdown
        spending_chart_label = QLabel("Spending Allocation")
        spending_chart_label.setStyleSheet("font-weight: bold;")
        charts_layout.addWidget(spending_chart_label)
        
        self.spending_chart = MacroStatBarChart(width=5, height=3)
        charts_layout.addWidget(self.spending_chart)
        
        right_layout.addWidget(charts_group)
        
        # Add the widgets to the splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        
        # Set initial sizes
        splitter.setSizes([600, 800])
        
        layout.addWidget(splitter)
        
        self.tabs.addTab(govt_tab, "Government")
        
    def create_central_bank_tab(self):
        """Create the central bank/monetary policy tab"""
        cb_tab = QWidget()
        layout = QVBoxLayout(cb_tab)
        
        # Split into two columns
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Monetary tools and policy controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Interest rate controls
        interest_group = QGroupBox("Monetary Policy Controls")
        interest_layout = QVBoxLayout(interest_group)
        
        # Current policy stance
        stance_widget = QWidget()
        stance_layout = QHBoxLayout(stance_widget)
        
        stance_label = QLabel("Current Policy Stance:")
        stance_label.setStyleSheet("font-weight: bold;")
        stance_layout.addWidget(stance_label)
        
        self.stance_value = QLabel(self.economy.central_bank.policy_stance)
        stance_layout.addWidget(self.stance_value)
        stance_layout.addStretch()
        
        interest_layout.addWidget(stance_widget)
        
        # Interest rate slider
        rate_label = QLabel("Base Interest Rate:")
        rate_label.setStyleSheet(f"font-weight: bold; color: {Config.COLORS['primary']};")
        interest_layout.addWidget(rate_label)
        
        rate_widget = QWidget()
        rate_layout = QHBoxLayout(rate_widget)
        rate_layout.setContentsMargins(0, 0, 0, 0)
        
        self.interest_slider = ModernSlider(Qt.Orientation.Horizontal)
        self.interest_slider.setMinimum(0)
        self.interest_slider.setMaximum(200)  # 0% to 20%
        current_rate = int(self.economy.central_bank.interest_rate * 1000)
        self.interest_slider.setValue(current_rate)
        rate_layout.addWidget(self.interest_slider, 4)
        
        self.interest_value = QLabel(f"{self.economy.central_bank.interest_rate*100:.2f}%")
        rate_layout.addWidget(self.interest_value, 1)
        
        self.interest_slider.valueChanged.connect(lambda val: self.interest_value.setText(f"{val/10:.2f}%"))
        
        interest_layout.addWidget(rate_widget)
        
        # Inflation target
        target_label = QLabel("Inflation Target:")
        interest_layout.addWidget(target_label)
        
        target_widget = QWidget()
        target_layout = QHBoxLayout(target_widget)
        target_layout.setContentsMargins(0, 0, 0, 0)
        
        self.inflation_slider = ModernSlider(Qt.Orientation.Horizontal)
        self.inflation_slider.setMinimum(0)
        self.inflation_slider.setMaximum(100)  # 0% to 10%
        current_target = int(self.economy.central_bank.inflation_target * 1000)
        self.inflation_slider.setValue(current_target)
        target_layout.addWidget(self.inflation_slider, 4)
        
        self.inflation_value = QLabel(f"{self.economy.central_bank.inflation_target*100:.2f}%")
        target_layout.addWidget(self.inflation_value, 1)
        
        self.inflation_slider.valueChanged.connect(lambda val: self.inflation_value.setText(f"{val/10:.2f}%"))
        
        interest_layout.addWidget(target_widget)
        
        # Reserve ratio
        reserve_label = QLabel("Reserve Ratio:")
        interest_layout.addWidget(reserve_label)
        
        reserve_widget = QWidget()
        reserve_layout = QHBoxLayout(reserve_widget)
        reserve_layout.setContentsMargins(0, 0, 0, 0)
        
        self.reserve_slider = ModernSlider(Qt.Orientation.Horizontal)
        self.reserve_slider.setMinimum(0)
        self.reserve_slider.setMaximum(500)  # 0% to 50%
        current_reserve = int(self.economy.central_bank.reserve_ratio * 1000)
        self.reserve_slider.setValue(current_reserve)
        reserve_layout.addWidget(self.reserve_slider, 4)
        
        self.reserve_value = QLabel(f"{self.economy.central_bank.reserve_ratio*100:.2f}%")
        reserve_layout.addWidget(self.reserve_value, 1)
        
        self.reserve_slider.valueChanged.connect(lambda val: self.reserve_value.setText(f"{val/10:.2f}%"))
        
        interest_layout.addWidget(reserve_widget)
        
        # Apply button
        apply_button = ModernPushButton("Apply Monetary Policy Changes", True)
        apply_button.clicked.connect(self.apply_monetary_policy)
        interest_layout.addWidget(apply_button)
        
        left_layout.addWidget(interest_group)
        
        # Money supply controls
        money_group = QGroupBox("Money Supply")
        money_layout = QVBoxLayout(money_group)
        
        # Current money supply
        money_widget = QWidget()
        money_layout_inner = QHBoxLayout(money_widget)
        
        money_label = QLabel("Current Money Supply:")
        money_label.setStyleSheet("font-weight: bold;")
        money_layout_inner.addWidget(money_label)
        
        self.money_value = QLabel(format_currency(self.economy.central_bank.money_supply))
        money_layout_inner.addWidget(self.money_value)
        money_layout_inner.addStretch()
        
        money_layout.addWidget(money_widget)
        
        # Money supply chart placeholder
        self.money_chart = MacroStatLineChart(width=5, height=3)
        money_layout.addWidget(self.money_chart)
        
        left_layout.addWidget(money_group)
        left_layout.addStretch()
        
        # Right side - Current status and effects
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Key indicators
        indicators_group = QGroupBox("Key Economic Indicators")
        indicators_layout = QVBoxLayout(indicators_group)
        
        # Current inflation
        inflation_widget = QWidget()
        inflation_layout = QHBoxLayout(inflation_widget)
        
        current_inflation_label = QLabel("Current Inflation Rate:")
        current_inflation_label.setStyleSheet("font-weight: bold;")
        inflation_layout.addWidget(current_inflation_label)
        
        self.current_inflation_value = QLabel(f"{self.economy.global_conditions['inflation']*100:.2f}%")
        inflation_layout.addWidget(self.current_inflation_value)
        
        # Show gap to target
        gap = self.economy.global_conditions["inflation"] - self.economy.central_bank.inflation_target
        gap_color = Config.COLORS["positive"] if abs(gap) < 0.005 else Config.COLORS["warning"] if abs(gap) < 0.02 else Config.COLORS["negative"]
        
        gap_label = QLabel(f"({'+' if gap > 0 else ''}{gap*100:.2f}% vs target)")
        gap_label.setStyleSheet(f"color: {gap_color};")
        inflation_layout.addWidget(gap_label)
        
        inflation_layout.addStretch()
        indicators_layout.addWidget(inflation_widget)
        
        # GDP growth
        growth_widget = QWidget()
        growth_layout = QHBoxLayout(growth_widget)
        
        growth_label = QLabel("GDP Growth Rate:")
        growth_label.setStyleSheet("font-weight: bold;")
        growth_layout.addWidget(growth_label)
        
        self.growth_value = QLabel(f"{self.economy.global_conditions['economic_growth']*100:.2f}%")
        growth_color = Config.COLORS["positive"] if self.economy.global_conditions["economic_growth"] > 0.01 else Config.COLORS["warning"] if self.economy.global_conditions["economic_growth"] > 0 else Config.COLORS["negative"]
        self.growth_value.setStyleSheet(f"color: {growth_color};")
        growth_layout.addWidget(self.growth_value)
        growth_layout.addStretch()
        
        indicators_layout.addWidget(growth_widget)
        
        # Unemployment
        unemployment_widget = QWidget()
        unemployment_layout = QHBoxLayout(unemployment_widget)
        
        unemployment_label = QLabel("Unemployment Rate:")
        unemployment_label.setStyleSheet("font-weight: bold;")
        unemployment_layout.addWidget(unemployment_label)
        
        self.unemployment_cb_value = QLabel(f"{self.economy.global_conditions['unemployment']*100:.2f}%")
        unemployment_color = Config.COLORS["positive"] if self.economy.global_conditions["unemployment"] < 0.05 else Config.COLORS["warning"] if self.economy.global_conditions["unemployment"] < 0.08 else Config.COLORS["negative"]
        self.unemployment_cb_value.setStyleSheet(f"color: {unemployment_color};")
        unemployment_layout.addWidget(self.unemployment_cb_value)
        unemployment_layout.addStretch()
        
        indicators_layout.addWidget(unemployment_widget)
        
        # Consumer confidence
        confidence_widget = QWidget()
        confidence_layout = QHBoxLayout(confidence_widget)
        
        confidence_label = QLabel("Consumer Confidence:")
        confidence_label.setStyleSheet("font-weight: bold;")
        confidence_layout.addWidget(confidence_label)
        
        self.confidence_value = QLabel(f"{self.economy.global_conditions['consumer_confidence']*100:.1f}%")
        confidence_color = Config.COLORS["positive"] if self.economy.global_conditions["consumer_confidence"] > 0.6 else Config.COLORS["warning"] if self.economy.global_conditions["consumer_confidence"] > 0.4 else Config.COLORS["negative"]
        self.confidence_value.setStyleSheet(f"color: {confidence_color};")
        confidence_layout.addWidget(self.confidence_value)
        confidence_layout.addStretch()
        
        indicators_layout.addWidget(confidence_widget)
        
        right_layout.addWidget(indicators_group)
        
        # Policy effectiveness
        effectiveness_group = QGroupBox("Policy Impact Analysis")
        effectiveness_layout = QVBoxLayout(effectiveness_group)
        
        # Chart showing interest rate vs inflation over time
        self.policy_chart = MacroStatLineChart(width=8, height=4)
        effectiveness_layout.addWidget(self.policy_chart)
        
        # Text analysis
        self.policy_analysis = QTextEdit()
        self.policy_analysis.setReadOnly(True)
        self.policy_analysis.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Config.COLORS["surface_light"]};
                color: {Config.COLORS["text_primary"]};
                border: 1px solid {Config.COLORS["border"]};
                border-radius: 4px;
                padding: 8px;
            }}
        """)
        self.policy_analysis.setMaximumHeight(150)
        effectiveness_layout.addWidget(self.policy_analysis)
        
        right_layout.addWidget(effectiveness_group)
        
        # Add the widgets to the splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        
        # Set initial sizes
        splitter.setSizes([600, 800])
        
        layout.addWidget(splitter)
        
        self.tabs.addTab(cb_tab, "Central Bank")
        
    def create_housing_market_tab(self):
        """Create the housing market tab"""
        housing_tab = QWidget()
        layout = QVBoxLayout(housing_tab)
        
        # Top statistics
        stats_layout = QHBoxLayout()
        
        self.avg_price_card = StatsCard("Average House Price", format_currency(self.economy.housing_market.avg_house_price), "National average")
        self.annual_growth_card = StatsCard("Annual Price Growth", "0.0%", "Year-on-year change")
        self.transactions_card = StatsCard("Monthly Transactions", "0", "Sales volume")
        self.rental_yield_card = StatsCard("Rental Yield", "0.0%", "Annual return")
        self.affordability_card = StatsCard("Affordability Ratio", "0.0", "Price to income")
        self.bubble_risk_card = StatsCard("Bubble Risk", "0.0%", "Risk rating")
        
        stats_layout.addWidget(self.avg_price_card)
        stats_layout.addWidget(self.annual_growth_card)
        stats_layout.addWidget(self.transactions_card)
        stats_layout.addWidget(self.rental_yield_card)
        stats_layout.addWidget(self.affordability_card)
        stats_layout.addWidget(self.bubble_risk_card)
        
        layout.addLayout(stats_layout)
        
        # Split into two sections
        content_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Price trends chart
        trends_widget = QWidget()
        trends_layout = QVBoxLayout(trends_widget)
        
        trends_group = QGroupBox("Housing Market Trends")
        trends_inner_layout = QVBoxLayout(trends_group)
        
        self.house_price_chart = MacroStatLineChart(width=10, height=4)
        trends_inner_layout.addWidget(self.house_price_chart)
        
        trends_layout.addWidget(trends_group)
        
        # Regional prices and comparison
        regional_group = QGroupBox("Regional Housing Market")
        regional_layout = QVBoxLayout(regional_group)
        
        self.regional_chart = MacroStatBarChart(width=10, height=3)
        regional_layout.addWidget(self.regional_chart)
        
        trends_layout.addWidget(regional_group)
        
        content_splitter.addWidget(trends_widget)
        
        # Bottom section - Housing market analysis
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        
        # Market factors
        factors_group = QGroupBox("Housing Market Factors")
        factors_layout = QHBoxLayout(factors_group)
        
        # Left column - economic factors
        economic_widget = QWidget()
        economic_layout = QVBoxLayout(economic_widget)
        
        economic_header = QLabel("Economic Factors")
        economic_header.setStyleSheet(f"font-weight: bold; color: {Config.COLORS['primary']};")
        economic_layout.addWidget(economic_header)
        
        # Interest rate impact
        interest_impact_widget = QWidget()
        interest_impact_layout = QHBoxLayout(interest_impact_widget)
        interest_impact_layout.setContentsMargins(0, 0, 0, 0)
        
        interest_impact_label = QLabel("Interest Rate Impact:")
        interest_impact_layout.addWidget(interest_impact_label)
        
        self.interest_impact_value = QLabel("0")
        interest_impact_layout.addWidget(self.interest_impact_value)
        interest_impact_layout.addStretch()
        
        economic_layout.addWidget(interest_impact_widget)
        
        # Economic growth impact
        growth_impact_widget = QWidget()
        growth_impact_layout = QHBoxLayout(growth_impact_widget)
        growth_impact_layout.setContentsMargins(0, 0, 0, 0)
        
        growth_impact_label = QLabel("Economic Growth Impact:")
        growth_impact_layout.addWidget(growth_impact_label)
        
        self.growth_impact_value = QLabel("0")
        growth_impact_layout.addWidget(self.growth_impact_value)
        growth_impact_layout.addStretch()
        
        economic_layout.addWidget(growth_impact_widget)
        
        # Unemployment impact
        unemployment_impact_widget = QWidget()
        unemployment_impact_layout = QHBoxLayout(unemployment_impact_widget)
        unemployment_impact_layout.setContentsMargins(0, 0, 0, 0)
        
        unemployment_impact_label = QLabel("Unemployment Impact:")
        unemployment_impact_layout.addWidget(unemployment_impact_label)
        
        self.unemployment_impact_value = QLabel("0")
        unemployment_impact_layout.addWidget(self.unemployment_impact_value)
        unemployment_impact_layout.addStretch()
        
        economic_layout.addWidget(unemployment_impact_widget)
        
        # Consumer confidence impact
        confidence_impact_widget = QWidget()
        confidence_impact_layout = QHBoxLayout(confidence_impact_widget)
        confidence_impact_layout.setContentsMargins(0, 0, 0, 0)
        
        confidence_impact_label = QLabel("Consumer Confidence Impact:")
        confidence_impact_layout.addWidget(confidence_impact_label)
        
        self.confidence_impact_value = QLabel("0")
        confidence_impact_layout.addWidget(self.confidence_impact_value)
        confidence_impact_layout.addStretch()
        
        economic_layout.addWidget(confidence_impact_widget)
        
        factors_layout.addWidget(economic_widget)
        
        # Right column - market indicators
        market_widget = QWidget()
        market_layout = QVBoxLayout(market_widget)
        
        market_header = QLabel("Market Indicators")
        market_header.setStyleSheet(f"font-weight: bold; color: {Config.COLORS['primary']};")
        market_layout.addWidget(market_header)
        
        # Supply-demand balance
        supply_demand_widget = QWidget()
        supply_demand_layout = QHBoxLayout(supply_demand_widget)
        supply_demand_layout.setContentsMargins(0, 0, 0, 0)
        
        supply_demand_label = QLabel("Supply-Demand Balance:")
        supply_demand_layout.addWidget(supply_demand_label)
        
        self.supply_demand_value = QLabel("0")
        supply_demand_layout.addWidget(self.supply_demand_value)
        supply_demand_layout.addStretch()
        
        market_layout.addWidget(supply_demand_widget)
        
        # Price-to-income ratio
        price_income_widget = QWidget()
        price_income_layout = QHBoxLayout(price_income_widget)
        price_income_layout.setContentsMargins(0, 0, 0, 0)
        
        price_income_label = QLabel("Price-to-Income Ratio:")
        price_income_layout.addWidget(price_income_label)
        
        self.price_income_value = QLabel("0")
        price_income_layout.addWidget(self.price_income_value)
        price_income_layout.addStretch()
        
        market_layout.addWidget(price_income_widget)
        
        # Rental yield
        rental_yield_widget = QWidget()
        rental_yield_layout = QHBoxLayout(rental_yield_widget)
        rental_yield_layout.setContentsMargins(0, 0, 0, 0)
        
        rental_yield_label = QLabel("Rental Yield:")
        rental_yield_layout.addWidget(rental_yield_label)
        
        self.rental_yield_detail_value = QLabel("0")
        rental_yield_layout.addWidget(self.rental_yield_detail_value)
        rental_yield_layout.addStretch()
        
        market_layout.addWidget(rental_yield_widget)
        
        # Bubble risk
        bubble_widget = QWidget()
        bubble_layout = QHBoxLayout(bubble_widget)
        bubble_layout.setContentsMargins(0, 0, 0, 0)
        
        bubble_label = QLabel("Housing Bubble Risk:")
        bubble_layout.addWidget(bubble_label)
        
        self.bubble_value = QLabel("0")
        bubble_layout.addWidget(self.bubble_value)
        bubble_layout.addStretch()
        
        market_layout.addWidget(bubble_widget)
        
        factors_layout.addWidget(market_widget)
        
        analysis_layout.addWidget(factors_group)
        
        # Market analysis
        analysis_group = QGroupBox("Market Analysis")
        analysis_inner_layout = QVBoxLayout(analysis_group)
        
        self.housing_analysis = QTextEdit()
        self.housing_analysis.setReadOnly(True)
        self.housing_analysis.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Config.COLORS["surface_light"]};
                color: {Config.COLORS["text_primary"]};
                border: 1px solid {Config.COLORS["border"]};
                border-radius: 4px;
                padding: 8px;
            }}
        """)
        analysis_inner_layout.addWidget(self.housing_analysis)
        
        analysis_layout.addWidget(analysis_group)
        
        content_splitter.addWidget(analysis_widget)
        
        # Set initial sizes
        content_splitter.setSizes([400, 300])
        
        layout.addWidget(content_splitter)
        
        self.tabs.addTab(housing_tab, "Housing Market")
        
    def create_stock_market_tab(self):
        """Create the stock market tab"""
        stock_tab = QWidget()
        layout = QVBoxLayout(stock_tab)
        
        # Top statistics cards
        stats_layout = QHBoxLayout()
        
        self.market_index_card = StatsCard("Market Index", "0", "Main stock index")
        self.market_return_card = StatsCard("Monthly Return", "0.0%", "Current month")
        self.annual_market_return_card = StatsCard("Annual Return", "0.0%", "Year-on-year")
        self.market_sentiment_card = StatsCard("Market Sentiment", "0.0%", "Bull/bear indicator")
        self.volatility_card = StatsCard("Volatility", "0.0%", "Price variation")
        self.public_companies_card = StatsCard("Listed Companies", "0", "Public companies count")
        
        stats_layout.addWidget(self.market_index_card)
        stats_layout.addWidget(self.market_return_card)
        stats_layout.addWidget(self.annual_market_return_card)
        stats_layout.addWidget(self.market_sentiment_card)
        stats_layout.addWidget(self.volatility_card)
        stats_layout.addWidget(self.public_companies_card)
        
        layout.addLayout(stats_layout)
        
        # Split the main area
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Market index and trends
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Market index chart
        index_group = QGroupBox("Market Index")
        index_layout = QVBoxLayout(index_group)
        
        self.index_chart = MacroStatLineChart(width=6, height=4)
        index_layout.addWidget(self.index_chart)
        
        left_layout.addWidget(index_group)
        
        # Market sentiment chart
        sentiment_group = QGroupBox("Market Sentiment & Volatility")
        sentiment_layout = QVBoxLayout(sentiment_group)
        
        self.sentiment_chart = MacroStatLineChart(width=6, height=4)
        sentiment_layout.addWidget(self.sentiment_chart)
        
        left_layout.addWidget(sentiment_group)
        
        # Right side - Stock listings and company details
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Filter controls
        filter_widget = QWidget()
        filter_layout = QHBoxLayout(filter_widget)
        
        filter_label = QLabel("Filter by:")
        filter_layout.addWidget(filter_label)
        
        self.stock_sector_filter = ModernComboBox()
        self.stock_sector_filter.addItems(["All Sectors"] + Config.BUSINESS_SECTORS)
        filter_layout.addWidget(self.stock_sector_filter)
        
        sort_label = QLabel("Sort by:")
        filter_layout.addWidget(sort_label)
        
        self.stock_sort = ModernComboBox()
        self.stock_sort.addItems(["Market Cap", "Price", "Daily Change", "Company Name"])
        filter_layout.addWidget(self.stock_sort)
        
        filter_layout.addStretch()
        
        right_layout.addWidget(filter_widget)
        
        # Stock listings table
        self.stock_table = QTableWidget()
        self.stock_table.setColumnCount(7)
        self.stock_table.setHorizontalHeaderLabels([
            "Company", "Price", "Change", "Market Cap", 
            "P/E Ratio", "Revenue", "Sector"
        ])
        self.stock_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)  # Company name
        
        right_layout.addWidget(self.stock_table)
        
        # Selected stock chart
        selected_group = QGroupBox("Selected Stock")
        selected_layout = QVBoxLayout(selected_group)
        
        # Company selector
        company_layout = QHBoxLayout()
        
        company_label = QLabel("Select Company:")
        company_layout.addWidget(company_label)
        
        self.stock_company_selector = ModernComboBox()
        self.stock_company_selector.addItem("Select a company...")
        company_layout.addWidget(self.stock_company_selector)
        company_layout.addStretch()
        
        selected_layout.addLayout(company_layout)
        
        # Stock price chart
        self.stock_price_chart = MacroStatLineChart(width=6, height=3)
        selected_layout.addWidget(self.stock_price_chart)
        
        right_layout.addWidget(selected_group)
        
        # Add widgets to the splitter
        content_splitter.addWidget(left_widget)
        content_splitter.addWidget(right_widget)
        
        # Set initial sizes
        content_splitter.setSizes([700, 700])
        
        layout.addWidget(content_splitter)
        
        self.tabs.addTab(stock_tab, "Stock Market")
        
    def create_statistics_tab(self):
        """Create the statistics/data tab"""
        stats_tab = QWidget()
        layout = QVBoxLayout(stats_tab)
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        category_label = QLabel("Category:")
        controls_layout.addWidget(category_label)
        
        self.stats_category = ModernComboBox()
        self.stats_category.addItems([
            "GDP & Growth", "Employment & Wages", "Inflation & Prices",
            "Government & Debt", "Financial Markets", "Demographics",
            "Business & Industry", "Housing & Real Estate"
        ])
        self.stats_category.currentIndexChanged.connect(self.update_statistics_metric)
        controls_layout.addWidget(self.stats_category)
        
        metric_label = QLabel("Metric:")
        controls_layout.addWidget(metric_label)
        
        self.stats_metric = ModernComboBox()
        controls_layout.addWidget(self.stats_metric)
        
        timeframe_label = QLabel("Timeframe:")
        controls_layout.addWidget(timeframe_label)
        
        self.stats_timeframe = ModernComboBox()
        self.stats_timeframe.addItems(["All data", "Last 12 months", "Last 24 months", "Last 36 months"])
        controls_layout.addWidget(self.stats_timeframe)
        
        controls_layout.addStretch()
        
        update_button = ModernPushButton("Update Chart", True)
        update_button.clicked.connect(self.update_statistics_chart)
        controls_layout.addWidget(update_button)
        
        layout.addLayout(controls_layout)
        
        # Main chart area
        self.stats_chart = MacroStatLineChart(width=10, height=5)
        layout.addWidget(self.stats_chart)
        
        # Bottom section - Data table and export
        bottom_layout = QHBoxLayout()
        
        # Data table
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Date", "Value"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        bottom_layout.addWidget(self.stats_table, 3)
        
        # Export panel
        export_widget = QWidget()
        export_layout = QVBoxLayout(export_widget)
        
        export_group = QGroupBox("Export Data")
        export_inner_layout = QVBoxLayout(export_group)
        
        export_format_label = QLabel("Format:")
        export_inner_layout.addWidget(export_format_label)
        
        self.export_format = ModernComboBox()
        self.export_format.addItems(["CSV", "Excel", "JSON", "HTML"])
        export_inner_layout.addWidget(self.export_format)
        
        range_label = QLabel("Data Range:")
        export_inner_layout.addWidget(range_label)
        
        self.export_range = ModernComboBox()
        self.export_range.addItems(["Current metric only", "All metrics (selected category)", "All economic data"])
        export_inner_layout.addWidget(self.export_range)
        
        export_button = ModernPushButton("Export", True)
        export_button.clicked.connect(self.export_statistics)
        export_inner_layout.addWidget(export_button)
        
        export_layout.addWidget(export_group)
        export_layout.addStretch()
        
        bottom_layout.addWidget(export_widget, 1)
        
        layout.addLayout(bottom_layout, 1)
        
        # Initialize the first category
        self.update_statistics_metric(0)
        
        self.tabs.addTab(stats_tab, "Statistics")
        
    def create_events_tab(self):
        """Create the economic events tab"""
        events_tab = QWidget()
        layout = QVBoxLayout(events_tab)
        
        # Top filters
        filters_layout = QHBoxLayout()
        
        type_label = QLabel("Event Type:")
        filters_layout.addWidget(type_label)
        
        self.event_type_filter = ModernComboBox()
        self.event_type_filter.addItems([
            "All Events", "Economic Policy", "Market Events", "Housing Market", 
            "Central Bank", "Government", "External Shocks"
        ])
        filters_layout.addWidget(self.event_type_filter)
        
        severity_label = QLabel("Severity:")
        filters_layout.addWidget(severity_label)
        
        self.event_severity_filter = ModernComboBox()
        self.event_severity_filter.addItems(["All", "Low", "Medium", "High", "Critical"])
        filters_layout.addWidget(self.event_severity_filter)
        
        date_label = QLabel("Date Range:")
        filters_layout.addWidget(date_label)
        
        self.event_date_filter = ModernComboBox()
        self.event_date_filter.addItems(["All Time", "Last 12 Months", "Last 6 Months", "Last 3 Months"])
        filters_layout.addWidget(self.event_date_filter)
        
        filters_layout.addStretch()
        
        refresh_button = ModernPushButton("Refresh", False)
        refresh_button.clicked.connect(self.update_events_list)
        filters_layout.addWidget(refresh_button)
        
        layout.addLayout(filters_layout)
        
        # Main content - split view
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Events list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.events_table = QTableWidget()
        self.events_table.setColumnCount(4)
        self.events_table.setHorizontalHeaderLabels(["Date", "Type", "Severity", "Description"])
        self.events_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.events_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.events_table.selectionModel().selectionChanged.connect(self.event_selection_changed)
        
        left_layout.addWidget(self.events_table)
        
        # Right side - Event details and impact
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Event details
        details_group = QGroupBox("Event Details")
        details_layout = QVBoxLayout(details_group)
        
        self.event_title = QLabel("Select an event to view details")
        self.event_title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {Config.COLORS['primary']};")
        details_layout.addWidget(self.event_title)
        
        self.event_date = QLabel("")
        details_layout.addWidget(self.event_date)
        
        self.event_type = QLabel("")
        details_layout.addWidget(self.event_type)
        
        self.event_severity = QLabel("")
        details_layout.addWidget(self.event_severity)
        
        details_layout.addSpacing(10)
        
        self.event_description = QTextEdit()
        self.event_description.setReadOnly(True)
        self.event_description.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Config.COLORS["surface_light"]};
                color: {Config.COLORS["text_primary"]};
                border: 1px solid {Config.COLORS["border"]};
                border-radius: 4px;
                padding: 8px;
            }}
        """)
        details_layout.addWidget(self.event_description)
        
        right_layout.addWidget(details_group)
        
        # Economic impact
        impact_group = QGroupBox("Economic Impact")
        impact_layout = QVBoxLayout(impact_group)
        
        self.event_impact_chart = MacroStatLineChart(width=5, height=3)
        impact_layout.addWidget(self.event_impact_chart)
        
        self.event_impact_analysis = QTextEdit()
        self.event_impact_analysis.setReadOnly(True)
        self.event_impact_analysis.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Config.COLORS["surface_light"]};
                color: {Config.COLORS["text_primary"]};
                border: 1px solid {Config.COLORS["border"]};
                border-radius: 4px;
                padding: 8px;
            }}
        """)
        impact_layout.addWidget(self.event_impact_analysis)
        
        right_layout.addWidget(impact_group)
        
        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        
        # Set initial sizes
        splitter.setSizes([600, 600])
        
        layout.addWidget(splitter)
        
        self.tabs.addTab(events_tab, "Economic Events")
        
    def update_status_bar(self):
        """Update the status bar with current information"""
        try:
            stats = self.economy.get_summary_stats()
            self.statusBar.showMessage(
                f"Date: {stats.get('date', 'N/A')} | "
                f"GDP: {stats.get('gdp', 'N/A')} | "
                f"Unemployment: {stats.get('unemployment', 'N/A')} | "
                f"Inflation: {stats.get('inflation', 'N/A')} | "
                f"Population: {stats.get('population', 'N/A')}"
            )
        except Exception as e:
            print(f"Error updating status bar: {e}")
            self.statusBar.showMessage("Economic data loading...")
            
    def periodic_update(self):
        """Periodic UI updates"""
        # Update toolbar information
        stats = self.economy.get_summary_stats()
        self.date_label.setText(f"Date: {stats['date']}")
        self.gdp_label.setText(f"GDP: {stats['gdp']}")
        self.unemployment_label.setText(f"Unemployment: {stats['unemployment']}")
        self.inflation_label.setText(f"Inflation: {stats['inflation']}")
        self.interest_rate_label.setText(f"Interest Rate: {stats['interest_rate']}")
        
        # Update dashboard
        self.update_dashboard()
        
        # Update status bar
        self.update_status_bar()
        
    def run_simulation(self):
        """Run a simulation cycle"""
        # Disable the run button during simulation
        self.run_button.setEnabled(False)
        self.run_button.setText("Simulating...")
        
        # Create and run the simulation thread
        self.simulation_thread = SimulationThread(self.economy)
        self.simulation_thread.simulation_complete.connect(self.simulation_finished)
        self.simulation_thread.start()
        
    def simulation_finished(self):
        """Handle simulation completion"""
        # Re-enable the run button
        self.run_button.setEnabled(True)
        self.run_button.setText("Run Month")
        
        # Update UI with new data
        self.update_all_tabs()
        
    def update_all_tabs(self):
        """Update all tabs with current data"""
        self.update_dashboard()
        self.update_people_table()
        self.update_companies_table()
        self.update_government_tab()
        self.update_central_bank_tab()
        self.update_housing_market_tab()
        self.update_stock_market_tab()
        self.update_events_list()
        
    def update_dashboard(self):
        """Update the dashboard with current data"""
        stats = self.economy.get_summary_stats()
        
        # Update stats cards
        self.gdp_card.findChild(QLabel, "").setText(stats["gdp"])
        self.gdp_growth_card.findChild(QLabel, "").setText(stats["gdp_growth"])
        self.unemployment_card.findChild(QLabel, "").setText(stats["unemployment"])
        self.inflation_card.findChild(QLabel, "").setText(stats["inflation"])
        self.housing_card.findChild(QLabel, "").setText(format_currency(self.economy.housing_market.avg_house_price))
        self.stock_card.findChild(QLabel, "").setText(str(round(self.economy.stock_market.market_index, 1)))
        
        # Update GDP components chart if we have enough data
        if len(self.economy.macro_history["GDP"]) > 0:
            # GDP components (simplified)
            try:
                consumption = self.economy.macro_history["Consumption"][-1] * 12  # Annualized
            except:
                consumption = 0.0
            investment = self.economy.macro_history["Private Investment"][-1] * 12  # Annualized
            government = self.economy.macro_history["Government Spending"][-1] * 12  # Annualized
            net_exports = self.economy.macro_history["Trade Balance"][-1] * 12  # Annualized
            
            components = ["Consumption", "Investment", "Government", "Net Exports"]
            values = [consumption, investment, government, net_exports]
            
            self.gdp_chart.plot_bars(
                components, values, 
                title="GDP Components (Annual)",
                xlabel="Component", 
                ylabel="Value (£)"
            )
            
        # Update trend lines if we have enough data
        if len(self.economy.macro_history["GDP"]) > 1:
            # Get last 24 months or all available data
            months = min(24, len(self.economy.macro_history["GDP"]))
            x_values = list(range(months))
            
            # Plot multiple trends
            y_values_dict = {
                "GDP Growth": [self.economy.macro_history["GDP Growth"][-months:][i] * 100 for i in range(months)],
                "Unemployment": [self.economy.macro_history["Unemployment"][-months:][i] * 100 for i in range(months)],
                "Inflation": [self.economy.macro_history["Inflation"][-months:][i] * 100 for i in range(months)]
            }
            
            self.trends_chart.plot_multi_line(
                x_values, y_values_dict,
                title="Key Economic Indicators",
                xlabel="Month",
                ylabel="Percent (%)"
            )
            
        # Update recent events list
        self.events_list.clear()
        for event in self.economy.events[-10:]:  # Show latest 10 events
            date_str = event["date"].strftime("%b %Y")
            item_text = f"{date_str} - {event['type']}: {event['description'][:100]}..."
            
            item = QListWidgetItem(item_text)
            
            # Color code by severity
            if event["severity"] == "High":
                item.setForeground(QBrush(QColor(Config.COLORS["negative"])))
            elif event["severity"] == "Medium":
                item.setForeground(QBrush(QColor(Config.COLORS["warning"])))
            else:
                item.setForeground(QBrush(QColor(Config.COLORS["text_primary"])))
                
            self.events_list.insertItem(0, item)
            
    def update_people_table(self):
        """Update the people table with current data"""
        # Get filter and sort options
        filter_option = self.people_filter.currentText()
        sort_option = self.people_sort.currentText()
        search_text = self.people_search.text().lower()
        
        # Filter persons
        filtered_persons = []
        for person in self.economy.persons:
            # Apply search filter
            if search_text and search_text not in person.name.lower() and str(person.id).lower() not in search_text:
                continue
                
            # Apply type filter
            if filter_option == "Employed" and person.employment_status != "Employed":
                continue
            if filter_option == "Unemployed" and person.employment_status != "Unemployed":
                continue
            if filter_option == "Business Owners" and not person.is_business_owner:
                continue
            if filter_option == "High Net Worth" and person.net_worth < 1000000:
                continue
                
            filtered_persons.append(person)
            
        # Sort persons
        if sort_option == "Income (High to Low)":
            filtered_persons.sort(key=lambda p: p.monthly_income, reverse=True)
        elif sort_option == "Income (Low to High)":
            filtered_persons.sort(key=lambda p: p.monthly_income)
        elif sort_option == "Net Worth (High to Low)":
            filtered_persons.sort(key=lambda p: p.net_worth, reverse=True)
        elif sort_option == "Net Worth (Low to High)":
            filtered_persons.sort(key=lambda p: p.net_worth)
        elif sort_option == "Age (High to Low)":
            filtered_persons.sort(key=lambda p: p.age, reverse=True)
        elif sort_option == "Age (Low to High)":
            filtered_persons.sort(key=lambda p: p.age)
            
        # Calculate pagination
        persons_per_page = 20
        total_pages = (len(filtered_persons) + persons_per_page - 1) // persons_per_page
        self.people_page.setMaximum(max(1, total_pages))
        current_page = min(self.people_page.value(), total_pages)
        
        start_idx = (current_page - 1) * persons_per_page
        end_idx = min(start_idx + persons_per_page, len(filtered_persons))
        
        # Get current page of persons
        display_persons = filtered_persons[start_idx:end_idx]
        
        # Update table
        self.people_table.setRowCount(len(display_persons))
        
        for i, person in enumerate(display_persons):
            # ID
            id_item = QTableWidgetItem(str(person.id))
            self.people_table.setItem(i, 0, id_item)
            
            # Name
            name_item = QTableWidgetItem(person.name)
            self.people_table.setItem(i, 1, name_item)
            
            # Age
            age_item = QTableWidgetItem(str(person.age))
            self.people_table.setItem(i, 2, age_item)
            
            # Education
            education_item = QTableWidgetItem(person.education_level)
            self.people_table.setItem(i, 3, education_item)
            
            # Income
            income_item = QTableWidgetItem(format_currency(person.monthly_income * 12))  # Annual
            self.people_table.setItem(i, 4, income_item)
            
            # Net Worth
            net_worth_item = QTableWidgetItem(format_currency(person.net_worth))
            self.people_table.setItem(i, 5, net_worth_item)
            
            # Employment
            employment_item = QTableWidgetItem(person.employment_status)
            self.people_table.setItem(i, 6, employment_item)
            
            # Job
            job_item = QTableWidgetItem(person.job_title if person.job_title else "N/A")
            self.people_table.setItem(i, 7, job_item)
            
            # Housing
            housing_item = QTableWidgetItem(person.housing_status)
            self.people_table.setItem(i, 8, housing_item)
            
            # Happiness
            happiness_item = QTableWidgetItem(f"{person.happiness*100:.1f}%")
            self.people_table.setItem(i, 9, happiness_item)
            
    def update_companies_table(self):
        """Update the companies table with current data"""
        # Get filter and sort options
        sector_filter = self.companies_sector_filter.currentText()
        status_filter = self.companies_status_filter.currentText()
        sort_option = self.companies_sort.currentText()
        search_text = self.companies_search.text().lower()
        
        # Filter companies
        filtered_companies = []
        for company in self.economy.companies:
            # Apply search filter
            if search_text and search_text not in company.name.lower() and str(company.id).lower() not in search_text:
                continue
                
            # Apply sector filter
            if sector_filter != "All" and company.sector != sector_filter:
                continue
                
            # Apply status filter
            if status_filter == "Public" and not company.is_public:
                continue
            if status_filter == "Private" and company.is_public:
                continue
            if status_filter == "Profitable" and company.profit <= 0:
                continue
            if status_filter == "Loss-Making" and company.profit >= 0:
                continue
                
            filtered_companies.append(company)
            
        # Sort companies
        if sort_option == "Revenue (High to Low)":
            filtered_companies.sort(key=lambda c: c.revenue, reverse=True)
        elif sort_option == "Profit (High to Low)":
            filtered_companies.sort(key=lambda c: c.profit, reverse=True)
        elif sort_option == "Employees (High to Low)":
            filtered_companies.sort(key=lambda c: len(c.employees), reverse=True)
        elif sort_option == "Value (High to Low)":
            filtered_companies.sort(key=lambda c: c.value, reverse=True)
        elif sort_option == "Growth (High to Low)":
            filtered_companies.sort(key=lambda c: c.growth_rate if hasattr(c, 'growth_rate') else 0, reverse=True)
            
        # Calculate pagination
        companies_per_page = 20
        total_pages = (len(filtered_companies) + companies_per_page - 1) // companies_per_page
        self.companies_page.setMaximum(max(1, total_pages))
        current_page = min(self.companies_page.value(), total_pages)
        
        start_idx = (current_page - 1) * companies_per_page
        end_idx = min(start_idx + companies_per_page, len(filtered_companies))
        
        # Get current page of companies
        display_companies = filtered_companies[start_idx:end_idx]
        
        # Update table
        self.companies_table.setRowCount(len(display_companies))
        
        for i, company in enumerate(display_companies):
            # ID
            id_item = QTableWidgetItem(str(company.id)[:8])
            self.companies_table.setItem(i, 0, id_item)
            
            # Name
            name_item = QTableWidgetItem(company.name)
            self.companies_table.setItem(i, 1, name_item)
            
            # Sector
            sector_item = QTableWidgetItem(company.sector)
            self.companies_table.setItem(i, 2, sector_item)
            
            # Revenue (annual)
            revenue_item = QTableWidgetItem(format_currency(company.revenue * 12))
            self.companies_table.setItem(i, 3, revenue_item)
            
            # Profit (annual)
            profit_item = QTableWidgetItem(format_currency(company.profit * 12))
            if company.profit < 0:
                profit_item.setForeground(QBrush(QColor(Config.COLORS["negative"])))
            self.companies_table.setItem(i, 4, profit_item)
            
            # Profit margin
            margin = company.profit / company.revenue if company.revenue > 0 else 0
            margin_item = QTableWidgetItem(f"{margin*100:.1f}%")
            if margin < 0:
                margin_item.setForeground(QBrush(QColor(Config.COLORS["negative"])))
            self.companies_table.setItem(i, 5, margin_item)
            
            # Employees
            employees_item = QTableWidgetItem(str(len(company.employees)))
            self.companies_table.setItem(i, 6, employees_item)
            
            # Value
            value_item = QTableWidgetItem(format_currency(company.value))
            self.companies_table.setItem(i, 7, value_item)
            
            # Public status
            public_item = QTableWidgetItem("Yes" if company.is_public else "No")
            self.companies_table.setItem(i, 8, public_item)
            
            # Growth rate
            growth = company.growth_rate if hasattr(company, 'growth_rate') else 0
            growth_item = QTableWidgetItem(f"{growth*100:.1f}%")
            if growth > 0:
                growth_item.setForeground(QBrush(QColor(Config.COLORS["positive"])))
            elif growth < 0:
                growth_item.setForeground(QBrush(QColor(Config.COLORS["negative"])))
            self.companies_table.setItem(i, 9, growth_item)

    def update_government_tab(self):
        """Update the government tab with current data"""
        # Update current fiscal status
        stats = self.economy.government.get_summary_stats()
        
        self.revenue_value.setText(format_currency(stats["tax_revenue"]))
        self.spending_value.setText(format_currency(stats["spending"]))
        self.balance_value.setText(format_currency(stats["budget_balance"]))
        self.debt_value.setText(format_currency(stats["debt"]))
        self.debt_gdp_value.setText(f"{stats['debt_to_gdp']*100:.1f}%")
        
        # Update tax revenue chart
        tax_categories = [
            "Income Tax", 
            "Corporate Tax", 
            "Sales Tax", 
            "National Insurance"
        ]
        
        tax_values = [
            stats["income_tax"],
            stats["corporation_tax"],
            stats["sales_tax"],
            stats["national_insurance"]
        ]
        
        self.tax_chart.plot_bars(
            tax_categories, tax_values,
            title="Annual Tax Revenue by Type",
            ylabel="Revenue (£)"
        )
        
        # Update spending chart
        spending_categories = list(self.economy.government.spending_allocation.keys())
        spending_values = [
            self.economy.government.spending_allocation[cat] * stats["spending"]
            for cat in spending_categories
        ]
        
        self.spending_chart.plot_bars(
            spending_categories, spending_values,
            title="Government Spending by Department",
            ylabel="Spending (£)"
        )
        
    def update_central_bank_tab(self):
        """Update the central bank tab with current data"""
        # Update policy stance
        self.stance_value.setText(self.economy.central_bank.policy_stance)
        
        # Update key indicators
        self.current_inflation_value.setText(f"{self.economy.global_conditions['inflation']*100:.2f}%")
        self.growth_value.setText(f"{self.economy.global_conditions['economic_growth']*100:.2f}%")
        self.unemployment_cb_value.setText(f"{self.economy.global_conditions['unemployment']*100:.2f}%")
        self.confidence_value.setText(f"{self.economy.global_conditions['consumer_confidence']*100:.1f}%")
        
        # Update money supply
        self.money_value.setText(format_currency(self.economy.central_bank.money_supply))
        
        # Update money supply chart
        if len(self.economy.central_bank.money_supply_history) > 1:
            months = min(24, len(self.economy.central_bank.money_supply_history))
            x_values = list(range(months))
            y_values = [self.economy.central_bank.money_supply_history[-months:][i] / 1e9 for i in range(months)]
            
            self.money_chart.plot_line(
                x_values, y_values,
                title="Money Supply Over Time",
                xlabel="Month",
                ylabel="Money Supply (£B)",
                color=Config.COLORS["chart_line"]
            )
            
        # Update policy impact chart
        if len(self.economy.central_bank.interest_rate_history) > 1:
            months = min(24, len(self.economy.central_bank.interest_rate_history))
            x_values = list(range(months))
            
            y_values_dict = {
                "Interest Rate": [self.economy.central_bank.interest_rate_history[-months:][i] * 100 for i in range(months)],
                "Inflation": [self.economy.central_bank.inflation_history[-months:][i] * 100 for i in range(months)]
            }
            
            self.policy_chart.plot_multi_line(
                x_values, y_values_dict,
                title="Interest Rate vs Inflation",
                xlabel="Month",
                ylabel="Percent (%)"
            )
            
        # Generate policy analysis
        self.generate_policy_analysis()
        
    def update_housing_market_tab(self):
        """Update the housing market tab with current data"""
        # Update stats cards
        self.avg_price_card.findChild(QLabel, "").setText(format_currency(self.economy.housing_market.avg_house_price))
        
        annual_growth = 0
        if len(self.economy.housing_market.price_history) >= 13:
            annual_growth = (self.economy.housing_market.price_history[-1] / self.economy.housing_market.price_history[-13] - 1) * 100
            self.annual_growth_card.findChild(QLabel, "").setText(f"{annual_growth:.1f}%")
            
        self.transactions_card.findChild(QLabel, "").setText(f"{int(self.economy.housing_market.monthly_transactions):,}")
        self.rental_yield_card.findChild(QLabel, "").setText(f"{self.economy.housing_market.rental_yield*100:.2f}%")
        
        # Affordability ratio (price to income)
        try:	
            affordability = self.economy.housing_market.avg_house_price / self.economy.avg_income
        except:
            affordability = 8.0
        self.affordability_card.findChild(QLabel, "").setText(f"{affordability:.1f}x")
        
        # Bubble risk
        self.bubble_risk_card.findChild(QLabel, "").setText(f"{self.economy.housing_market.bubble_factor*100:.1f}%")
        
        # Update price trends chart
        if len(self.economy.housing_market.price_history) > 1:
            months = min(36, len(self.economy.housing_market.price_history))
            x_values = list(range(months))
            y_values = [self.economy.housing_market.price_history[-months:][i] / 1000 for i in range(months)]
            
            self.house_price_chart.plot_line(
                x_values, y_values,
                title="Housing Prices Over Time",
                xlabel="Month",
                ylabel="Average House Price (£k)",
                color=Config.COLORS["chart_line"]
            )
            
        # Update regional chart
        regional_names = list(self.economy.housing_market.regional_prices.keys())
        regional_values = list(self.economy.housing_market.regional_prices.values())
        
        self.regional_chart.plot_bars(
            regional_names, regional_values,
            title="Regional Housing Prices",
            ylabel="Average Price (£)",
            color=Config.COLORS["primary"]
        )
        
        # Update market factors
        # Interest rate impact
        interest_impact = -0.5 * (self.economy.central_bank.interest_rate - 0.05)
        self.interest_impact_value.setText(
            f"{interest_impact:.2f}" +
            (" (Positive)" if interest_impact > 0 else " (Negative)" if interest_impact < 0 else " (Neutral)")
        )
        
        # Economic growth impact
        growth_impact = self.economy.global_conditions["economic_growth"] * 2
        self.growth_impact_value.setText(
            f"{growth_impact:.2f}" +
            (" (Positive)" if growth_impact > 0 else " (Negative)" if growth_impact < 0 else " (Neutral)")
        )
        
        # Unemployment impact
        unemployment_impact = -0.5 * (self.economy.global_conditions["unemployment"] - 0.05)
        self.unemployment_impact_value.setText(
            f"{unemployment_impact:.2f}" +
            (" (Positive)" if unemployment_impact > 0 else " (Negative)" if unemployment_impact < 0 else " (Neutral)")
        )
        
        # Consumer confidence impact
        confidence_impact = (self.economy.global_conditions["consumer_confidence"] - 0.5) * 0.5
        self.confidence_impact_value.setText(
            f"{confidence_impact:.2f}" +
            (" (Positive)" if confidence_impact > 0 else " (Negative)" if confidence_impact < 0 else " (Neutral)")
        )
        
        # Supply-demand balance (simplified)
        supply_demand = 0.5 + self.economy.housing_market.bubble_factor * 0.5
        self.supply_demand_value.setText(
            f"{supply_demand:.2f}" +
            (" (Demand > Supply)" if supply_demand > 0.6 else " (Supply > Demand)" if supply_demand < 0.4 else " (Balanced)")
        )
        
        # Price-to-income ratio
        try:
           price_income = self.economy.housing_market.avg_house_price / self.economy.avg_income
        except:
            price_income = 5
        self.price_income_value.setText(
            f"{price_income:.1f}x" +
            (" (High)" if price_income > 8 else " (Moderate)" if price_income > 5 else " (Low)")
        )
        
        # Rental yield
        self.rental_yield_detail_value.setText(f"{self.economy.housing_market.rental_yield*100:.2f}%")
        
        # Bubble risk
        bubble_risk = self.economy.housing_market.bubble_factor
        self.bubble_value.setText(
            f"{bubble_risk*100:.1f}%" +
            (" (High)" if bubble_risk > 0.7 else " (Moderate)" if bubble_risk > 0.4 else " (Low)")
        )
        
        # Generate housing market analysis
        self.generate_housing_analysis()
        
    def update_stock_market_tab(self):
        """Update the stock market tab with current data"""
        stats = self.economy.stock_market.get_summary_stats()
        
        # Update stats cards
        self.market_index_card.findChild(QLabel, "").setText(str(stats["market_index"]))
        self.market_return_card.findChild(QLabel, "").setText(stats["monthly_return"])
        self.annual_market_return_card.findChild(QLabel, "").setText(stats["annual_return"])
        self.market_sentiment_card.findChild(QLabel, "").setText(stats["sentiment"])
        self.volatility_card.findChild(QLabel, "").setText(stats["volatility"])
        self.public_companies_card.findChild(QLabel, "").setText(str(stats["public_companies"]))
        
        # Update market index chart
        if len(self.economy.stock_market.index_history) > 1:
            months = min(24, len(self.economy.stock_market.index_history))
            x_values = list(range(months))
            y_values = [self.economy.stock_market.index_history[-months:][i] for i in range(months)]
            
            self.index_chart.plot_line(
                x_values, y_values,
                title="Market Index",
                xlabel="Month",
                ylabel="Index Value",
                color=Config.COLORS["chart_line"]
            )
            
        # Update sentiment chart
        if len(self.economy.stock_market.sentiment_history) > 1:
            months = min(24, len(self.economy.stock_market.sentiment_history))
            x_values = list(range(months))
            
            y_values_dict = {
                "Sentiment": [self.economy.stock_market.sentiment_history[-months:][i] * 100 for i in range(months)],
                "Volatility": [self.economy.stock_market.volatility_history[-months:][i] * 100 for i in range(months)]
            }
            
            self.sentiment_chart.plot_multi_line(
                x_values, y_values_dict,
                title="Market Sentiment & Volatility",
                xlabel="Month",
                ylabel="Percent (%)"
            )
            
        # Update stock listings table
        self.update_stock_listings()
        
        # Update company selector dropdown
        self.update_stock_company_selector()
        
    def update_stock_listings(self):
        """Update the stock listings table with current data"""
        # Get filter and sort options
        sector_filter = self.stock_sector_filter.currentText()
        sort_option = self.stock_sort.currentText()
        
        # Filter companies
        filtered_companies = [c for c in self.economy.companies if c.is_public]
        
        if sector_filter != "All Sectors":
            filtered_companies = [c for c in filtered_companies if c.sector == sector_filter]
            
        # Sort companies
        if sort_option == "Market Cap":
            filtered_companies.sort(key=lambda c: c.value, reverse=True)
        elif sort_option == "Price":
            filtered_companies.sort(key=lambda c: c.share_price if hasattr(c, 'share_price') else 0, reverse=True)
        elif sort_option == "Daily Change":
            filtered_companies.sort(key=lambda c: (c.share_price / c.share_price_history[-2] - 1) if hasattr(c, 'share_price_history') and len(c.share_price_history) > 1 else 0, reverse=True)
        elif sort_option == "Company Name":
            filtered_companies.sort(key=lambda c: c.name)
            
        # Update table
        self.stock_table.setRowCount(len(filtered_companies))
        
        for i, company in enumerate(filtered_companies):
            # Company name
            name_item = QTableWidgetItem(company.name)
            self.stock_table.setItem(i, 0, name_item)
            
            # Share price
            price = company.share_price if hasattr(company, 'share_price') else 0
            price_item = QTableWidgetItem(format_currency(price))
            self.stock_table.setItem(i, 1, price_item)
            
            # Price change
            change = 0
            if hasattr(company, 'share_price_history') and len(company.share_price_history) > 1:
                change = (company.share_price / company.share_price_history[-2] - 1) * 100
                
            change_item = QTableWidgetItem(f"{change:+.2f}%")
            if change > 0:
                change_item.setForeground(QBrush(QColor(Config.COLORS["positive"])))
            elif change < 0:
                change_item.setForeground(QBrush(QColor(Config.COLORS["negative"])))
            self.stock_table.setItem(i, 2, change_item)
            
            # Market cap
            market_cap_item = QTableWidgetItem(format_currency(company.value))
            self.stock_table.setItem(i, 3, market_cap_item)
            
            # P/E ratio
            pe_ratio = 0
            if hasattr(company, 'profit') and company.profit > 0:
                pe_ratio = (company.share_price * company.shares_outstanding) / (company.profit * 12)
                
            pe_item = QTableWidgetItem(f"{pe_ratio:.1f}x" if pe_ratio > 0 else "N/A")
            self.stock_table.setItem(i, 4, pe_item)
            
            # Revenue
            revenue_item = QTableWidgetItem(format_currency(company.revenue * 12))  # Annual
            self.stock_table.setItem(i, 5, revenue_item)
            
            # Sector
            sector_item = QTableWidgetItem(company.sector)
            self.stock_table.setItem(i, 6, sector_item)
            
    def update_stock_company_selector(self):
        """Update the stock company selector dropdown"""
        current_selection = self.stock_company_selector.currentText()
        
        self.stock_company_selector.clear()
        self.stock_company_selector.addItem("Select a company...")
        
        for company in self.economy.companies:
            if company.is_public:
                self.stock_company_selector.addItem(company.name)
                
        # Try to restore previous selection
        index = self.stock_company_selector.findText(current_selection)
        if index >= 0:
            self.stock_company_selector.setCurrentIndex(index)
        else:
            self.stock_company_selector.setCurrentIndex(0)
            
        # Update the selected stock chart
        self.update_selected_stock_chart()
        
    def update_selected_stock_chart(self):
        """Update the selected stock price chart"""
        current_selection = self.stock_company_selector.currentText()
        
        if current_selection == "Select a company...":
            # Clear chart if no company selected
            self.stock_price_chart.figure.clear()
            self.stock_price_chart.draw()
            return
            
        # Find the selected company
        selected_company = None
        for company in self.economy.companies:
            if company.is_public and company.name == current_selection:
                selected_company = company
                break
                
        if selected_company and hasattr(selected_company, 'share_price_history') and len(selected_company.share_price_history) > 1:
            # Update the stock price chart
            months = len(selected_company.share_price_history)
            x_values = list(range(months))
            y_values = selected_company.share_price_history
            
            self.stock_price_chart.plot_line(
                x_values, y_values,
                title=f"{selected_company.name} Share Price",
                xlabel="Month",
                ylabel="Price (£)",
                color=Config.COLORS["chart_line"]
            )
        else:
            # Clear chart if no data available
            self.stock_price_chart.figure.clear()
            self.stock_price_chart.draw()
            
    def update_statistics_metric(self, category_index):
        """Update available metrics based on selected category"""
        self.stats_metric.clear()
        
        if category_index == 0:  # GDP & Growth
            metrics = [
                "GDP", "GDP Growth", "GDP Per Capita", 
                "Consumption", "Private Investment",
                "Government Spending", "Trade Balance"
            ]
        elif category_index == 1:  # Employment & Wages
            metrics = [
                "Unemployment", "Average Income", 
                "Median Income", "Gini Coefficient"
            ]
        elif category_index == 2:  # Inflation & Prices
            metrics = [
                "Inflation", "Consumer Confidence",
                "Housing Price", "Stock Market Index"
            ]
        elif category_index == 3:  # Government & Debt
            metrics = [
                "Government Debt", "Debt to GDP", 
                "Budget Balance", "Tax Revenue"
            ]
        elif category_index == 4:  # Financial Markets
            metrics = [
                "Stock Market Index", "Interest Rate", 
                "Money Supply", "Corporate Profits"
            ]
        elif category_index == 5:  # Demographics
            metrics = [
                "Population", "Business Count"
            ]
        elif category_index == 6:  # Business & Industry
            metrics = [
                "Business Count", "Corporate Profits", 
                "Private Investment"
            ]
        elif category_index == 7:  # Housing & Real Estate
            metrics = [
                "Housing Price", "Consumer Confidence"
            ]
        else:
            metrics = []
            
        self.stats_metric.addItems(metrics)
        
    def update_statistics_chart(self):
        """Update the statistics chart based on selected options"""
        category = self.stats_category.currentText()
        metric = self.stats_metric.currentText()
        timeframe = self.stats_timeframe.currentText()
        
        # Get data from economy history
        if metric in self.economy.macro_history and len(self.economy.macro_history[metric]) > 0:
            # Determine how many months to show
            if timeframe == "Last 12 months":
                months = min(12, len(self.economy.macro_history[metric]))
            elif timeframe == "Last 24 months":
                months = min(24, len(self.economy.macro_history[metric]))
            elif timeframe == "Last 36 months":
                months = min(36, len(self.economy.macro_history[metric]))
            else:
                months = len(self.economy.macro_history[metric])
                
            # Get data for selected timeframe
            x_values = list(range(1, months + 1))
            y_values = self.economy.macro_history[metric][-months:]
            
            # Format y-values for display
            formatted_y = []
            for val in y_values:
                if metric in ["GDP", "GDP Per Capita", "Tax Revenue", "Government Debt", "Government Spending", 
                             "Average Income", "Median Income", "Private Investment", "Consumption", "Housing Price",
                             "Money Supply", "Corporate Profits", "Trade Balance"]:
                    # Format as currency
                    formatted_y.append(format_currency(val, ""))
                elif metric in ["GDP Growth", "Unemployment", "Inflation", "Debt to GDP", "Budget Balance", "Gini Coefficient", 
                               "Consumer Confidence", "Interest Rate"]:
                    # Format as percentage
                    if metric == "Budget Balance":
                        # Format as percentage of GDP
                        gdp = self.economy.macro_history["GDP"][-months:][i] if i < len(self.economy.macro_history["GDP"]) else 1
                        formatted_y.append(f"{val/gdp*100:.2f}%")
                    else:
                        # Regular percentage
                        formatted_y.append(f"{val*100:.2f}%")
                else:
                    # Regular number
                    formatted_y.append(str(val))
                    
            # Update the chart
            self.stats_chart.plot_line(
                x_values, y_values,
                title=f"{metric} Over Time",
                xlabel="Month",
                ylabel=metric,
                color=Config.COLORS["chart_line"]
            )
            
            # Update the data table
            self.stats_table.setRowCount(months)
            
            for i in range(months):
                # Date (simplified)
                month_num = (self.economy.month - months + i + 1) % 12
                year_num = self.economy.current_date.year - ((self.economy.month - months + i) // 12)
                date_item = QTableWidgetItem(f"{month_num+1}/{year_num}")
                self.stats_table.setItem(i, 0, date_item)
                
                # Value
                value_item = QTableWidgetItem(str(y_values[i]))
                self.stats_table.setItem(i, 1, value_item)
        else:
            # Clear chart and table if no data available
            self.stats_chart.figure.clear()
            self.stats_chart.draw()
            self.stats_table.setRowCount(0)
            
    def export_statistics(self):
        """Export statistics data"""
        export_format = self.export_format.currentText()
        export_range = self.export_range.currentText()
        
        if export_format == "CSV":
            self.export_statistics_csv(export_range)
        elif export_format == "Excel":
            self.export_statistics_excel(export_range)
        elif export_format == "JSON":
            self.export_statistics_json(export_range)
        elif export_format == "HTML":
            self.export_statistics_html(export_range)
            
    def export_statistics_csv(self, export_range):
        """Export statistics to CSV file"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Statistics", "", "CSV Files (*.csv)"
            )
            
            if not filename:
                return
                
            if not filename.endswith('.csv'):
                filename += '.csv'
                
            # Determine which data to export
            data = {}
            if export_range == "Current metric only":
                metric = self.stats_metric.currentText()
                if metric in self.economy.macro_history:
                    data[metric] = self.economy.macro_history[metric]
            elif export_range == "All metrics (selected category)":
                category = self.stats_category.currentText()
                for metric in self.stats_metric.view().model().stringList():
                    if metric in self.economy.macro_history:
                        data[metric] = self.economy.macro_history[metric]
            else:  # All economic data
                data = self.economy.macro_history
                
            # Write to CSV
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                header = ['Month'] + list(data.keys())
                writer.writerow(header)
                
                # Get max length of any data series
                max_length = max(len(series) for series in data.values())
                
                # Write data rows
                for i in range(max_length):
                    row = [f"Month {i+1}"]
                    for metric in data.keys():
                        if i < len(data[metric]):
                            row.append(data[metric][i])
                        else:
                            row.append('')
                    writer.writerow(row)
                    
            QMessageBox.information(self, "Export Successful", 
                                   f"Statistics have been exported to {filename}")
            
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Error exporting data: {str(e)}")
            
    def export_statistics_excel(self, export_range):
        """Export statistics to Excel file"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Statistics", "", "Excel Files (*.xlsx)"
            )
            
            if not filename:
                return
                
            if not filename.endswith('.xlsx'):
                filename += '.xlsx'
                
            # Determine which data to export
            data = {}
            if export_range == "Current metric only":
                metric = self.stats_metric.currentText()
                if metric in self.economy.macro_history:
                    data[metric] = self.economy.macro_history[metric]
            elif export_range == "All metrics (selected category)":
                category = self.stats_category.currentText()
                for metric in self.stats_metric.view().model().stringList():
                    if metric in self.economy.macro_history:
                        data[metric] = self.economy.macro_history[metric]
            else:  # All economic data
                data = self.economy.macro_history
            
            # Create a pandas DataFrame
            df_dict = {'Month': [f"Month {i+1}" for i in range(max(len(series) for series in data.values()))]}
            
            for metric, series in data.items():
                df_dict[metric] = series + [''] * (len(df_dict['Month']) - len(series))
                
            df = pd.DataFrame(df_dict)
            
            # Export to Excel
            df.to_excel(filename, index=False)
            
            QMessageBox.information(self, "Export Successful", 
                                   f"Statistics have been exported to {filename}")
            
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Error exporting data: {str(e)}")
            
    def export_statistics_json(self, export_range):
        """Export statistics to JSON file"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Statistics", "", "JSON Files (*.json)"
            )
            
            if not filename:
                return
                
            if not filename.endswith('.json'):
                filename += '.json'
                
            # Determine which data to export
            data = {}
            if export_range == "Current metric only":
                metric = self.stats_metric.currentText()
                if metric in self.economy.macro_history:
                    data[metric] = self.economy.macro_history[metric]
            elif export_range == "All metrics (selected category)":
                category = self.stats_category.currentText()
                for metric in self.stats_metric.view().model().stringList():
                    if metric in self.economy.macro_history:
                        data[metric] = self.economy.macro_history[metric]
            else:  # All economic data
                data = self.economy.macro_history
                
            # Convert to serializable format (handle numpy types)
            serializable_data = {}
            for key, values in data.items():
                serializable_data[key] = [float(v) if isinstance(v, (np.float32, np.float64)) else v for v in values]
                
            # Write to JSON
            with open(filename, 'w') as jsonfile:
                json.dump(serializable_data, jsonfile, indent=2)
                
            QMessageBox.information(self, "Export Successful", 
                                   f"Statistics have been exported to {filename}")
            
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Error exporting data: {str(e)}")
            
    def export_statistics_html(self, export_range):
        """Export statistics to HTML file with interactive charts"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Statistics", "", "HTML Files (*.html)"
            )
            
            if not filename:
                return
                
            if not filename.endswith('.html'):
                filename += '.html'
                
            # Determine which data to export
            data = {}
            if export_range == "Current metric only":
                metric = self.stats_metric.currentText()
                if metric in self.economy.macro_history:
                    data[metric] = self.economy.macro_history[metric]
            elif export_range == "All metrics (selected category)":
                category = self.stats_category.currentText()
                for metric in self.stats_metric.view().model().stringList():
                    if metric in self.economy.macro_history:
                        data[metric] = self.economy.macro_history[metric]
            else:  # All economic data
                data = self.economy.macro_history
                
            # Create HTML with interactive charts using simple template
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>UltraEconomy Statistics - {datetime.now().strftime('%Y-%m-%d')}</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #121212; color: white; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .chart-container {{ width: 100%; height: 400px; margin-bottom: 30px; background-color: #1E1E1E; padding: 20px; border-radius: 8px; }}
                    h1, h2 {{ color: #01A9F4; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                    th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #333; }}
                    th {{ background-color: #1E1E1E; color: #01A9F4; }}
                    tr:nth-child(even) {{ background-color: #252525; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>UltraEconomy Statistics</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            """
            
            # Add a chart for each metric
            for i, (metric, values) in enumerate(data.items()):
                html_content += f"""
                    <div class="chart-container">
                        <h2>{metric}</h2>
                        <canvas id="chart{i}"></canvas>
                    </div>
                """
            
            # Add data tables for each metric
            html_content += "<h2>Data Tables</h2>"
            
            for metric, values in data.items():
                html_content += f"""
                    <h3>{metric}</h3>
                    <table>
                        <tr>
                            <th>Month</th>
                            <th>Value</th>
                        </tr>
                """
                
                for i, value in enumerate(values):
                    html_content += f"""
                        <tr>
                            <td>Month {i+1}</td>
                            <td>{value}</td>
                        </tr>
                    """
                    
                html_content += "</table>"
            
            # Add JavaScript for the charts
            html_content += """
                <script>
            """
            
            for i, (metric, values) in enumerate(data.items()):
                labels = [f"Month {j+1}" for j in range(len(values))]
                
                html_content += f"""
                    new Chart(document.getElementById('chart{i}'), {{
                        type: 'line',
                        data: {{
                            labels: {json.dumps(labels)},
                            datasets: [{{
                                label: '{metric}',
                                data: {json.dumps([float(v) if isinstance(v, (np.float32, np.float64)) else v for v in values])},
                                borderColor: '#01A9F4',
                                backgroundColor: 'rgba(1, 169, 244, 0.1)',
                                borderWidth: 2,
                                tension: 0.1
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                y: {{
                                    beginAtZero: false,
                                    grid: {{
                                        color: '#333'
                                    }},
                                    ticks: {{
                                        color: '#B0B0B0'
                                    }}
                                }},
                                x: {{
                                    grid: {{
                                        color: '#333'
                                    }},
                                    ticks: {{
                                        color: '#B0B0B0'
                                    }}
                                }}
                            }},
                            plugins: {{
                                legend: {{
                                    labels: {{
                                        color: '#FFFFFF'
                                    }}
                                }}
                            }}
                        }}
                    }});
                """
                
            html_content += """
                </script>
                </div>
            </body>
            </html>
            """
            
            # Write to file
            with open(filename, 'w') as htmlfile:
                htmlfile.write(html_content)
                
            QMessageBox.information(self, "Export Successful", 
                                   f"Statistics have been exported to {filename}")
            
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Error exporting data: {str(e)}")

    def update_events_list(self):
        """Update the events list with filtered events"""
        # Get filter options
        event_type = self.event_type_filter.currentText()
        severity = self.event_severity_filter.currentText()
        date_range = self.event_date_filter.currentText()
        
        # Filter events
        filtered_events = []
        for event in self.economy.events:
            # Apply event type filter
            if event_type != "All Events" and event_type not in event["type"]:
                continue
                
            # Apply severity filter
            if severity != "All" and severity != event["severity"]:
                continue
                
            # Apply date range filter
            if date_range != "All Time":
                months_ago = (self.economy.current_date - event["date"]).days / 30
                if date_range == "Last 3 Months" and months_ago > 3:
                    continue
                elif date_range == "Last 6 Months" and months_ago > 6:
                    continue
                elif date_range == "Last 12 Months" and months_ago > 12:
                    continue
                    
            filtered_events.append(event)
            
        # Update table
        self.events_table.setRowCount(len(filtered_events))
        
        for i, event in enumerate(filtered_events):
            # Date
            date_item = QTableWidgetItem(event["date"].strftime("%b %Y"))
            self.events_table.setItem(i, 0, date_item)
            
            # Type
            type_item = QTableWidgetItem(event["type"])
            self.events_table.setItem(i, 1, type_item)
            
            # Severity
            severity_item = QTableWidgetItem(event["severity"])
            severity_color = {
                "Low": QColor(Config.COLORS["text_primary"]),
                "Medium": QColor(Config.COLORS["warning"]),
                "High": QColor(Config.COLORS["negative"]),
                "Critical": QColor(Config.COLORS["danger"])
            }.get(event["severity"], QColor(Config.COLORS["text_primary"]))
            
            severity_item.setForeground(QBrush(severity_color))
            self.events_table.setItem(i, 2, severity_item)
            
            # Description (truncated)
            desc_item = QTableWidgetItem(event["description"][:100] + "..." if len(event["description"]) > 100 else event["description"])
            self.events_table.setItem(i, 3, desc_item)

    def event_selection_changed(self):
        """Handle event selection change in the events table"""
        selected_rows = self.events_table.selectionModel().selectedRows()
        if not selected_rows:
            return
            
        row = selected_rows[0].row()
        
        # Get event type and date from the table
        event_type = self.events_table.item(row, 1).text()
        event_date = self.events_table.item(row, 0).text()
        
        # Find the corresponding event in the economy
        for event in self.economy.events:
            if event["type"] == event_type and event["date"].strftime("%b %Y") == event_date:
                # Update the event details panel
                self.event_title.setText(event["type"])
                self.event_date.setText(f"Date: {event['date'].strftime('%B %Y')}")
                self.event_type.setText(f"Type: {event['type']}")
                self.event_severity.setText(f"Severity: {event['severity']}")
                
                # Set the severity text color
                severity_color = {
                    "Low": QColor(Config.COLORS["text_primary"]),
                    "Medium": QColor(Config.COLORS["warning"]),
                    "High": QColor(Config.COLORS["negative"]),
                    "Critical": QColor(Config.COLORS["danger"])
                }.get(event["severity"], QColor(Config.COLORS["text_primary"]))
                
                self.event_severity.setStyleSheet(f"color: {severity_color.name()};")
                
                # Set the full description
                self.event_description.setText(event["description"])
                
                # Generate and display economic impact analysis
                self.generate_event_impact_analysis(event)
                
                break

    def generate_event_impact_analysis(self, event):
        """Generate economic impact analysis for an event"""
        # Find data before and after the event
        event_date = event["date"]
        event_month = (event_date.year - Config.SIMULATION_START_YEAR) * 12 + (event_date.month - Config.SIMULATION_START_MONTH)
        
        # Check if we have enough data to analyze impact
        if event_month >= len(self.economy.macro_history["GDP"]) - 3:
            self.event_impact_analysis.setText("Insufficient data after the event to analyze impact.")
            self.event_impact_chart.figure.clear()
            self.event_impact_chart.draw()
            return
            
        # Analyze key indicators 3 months before and after event
        before_start = max(0, event_month - 3)
        before_end = event_month
        after_start = event_month + 1
        after_end = min(len(self.economy.macro_history["GDP"]), event_month + 4)
        
        # Plot impact on GDP, unemployment, and inflation
        indicators = ["GDP Growth", "Unemployment", "Inflation"]
        x_values = list(range(-3, 4))  # -3 to +3 months around event
        y_values_dict = {}
        
        for indicator in indicators:
            if indicator in self.economy.macro_history and len(self.economy.macro_history[indicator]) > before_start:
                values = []
                for i in range(-3, 4):
                    month_idx = event_month + i
                    if 0 <= month_idx < len(self.economy.macro_history[indicator]):
                        if indicator == "GDP Growth":
                            # GDP Growth is shown as percentage
                            values.append(self.economy.macro_history[indicator][month_idx] * 100)
                        elif indicator == "Unemployment":
                            # Unemployment is shown as percentage
                            values.append(self.economy.macro_history[indicator][month_idx] * 100)
                        elif indicator == "Inflation":
                            # Inflation is shown as percentage
                            values.append(self.economy.macro_history[indicator][month_idx] * 100)
                        else:
                            values.append(self.economy.macro_history[indicator][month_idx])
                    else:
                        values.append(None)  # No data available
                
                y_values_dict[indicator] = values
                
        # Plot impact chart
        self.event_impact_chart.figure.clear()
        ax = self.event_impact_chart.figure.add_subplot(111)
        ax.set_facecolor(Config.COLORS["background"])
        
        # Add vertical line at event
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # Plot each indicator
        for i, (indicator, values) in enumerate(y_values_dict.items()):
            ax.plot(x_values, values, marker='o', label=indicator, color=Config.CHART_COLORS[i])
            
        ax.set_title(f"Economic Impact of {event['type']}", color=Config.COLORS["text_primary"])
        ax.set_xlabel("Months from Event", color=Config.COLORS["text_secondary"])
        ax.set_ylabel("Value", color=Config.COLORS["text_secondary"])
        ax.grid(True, linestyle='--', alpha=0.3, color=Config.COLORS["chart_grid"])
        ax.tick_params(axis='x', colors=Config.COLORS["text_secondary"])
        ax.tick_params(axis='y', colors=Config.COLORS["text_secondary"])
        ax.legend(facecolor=Config.COLORS["surface_light"], edgecolor=Config.COLORS["border"], 
                 labelcolor=Config.COLORS["text_primary"])
                 
        self.event_impact_chart.draw()
        
        # Generate text analysis of impact
        analysis_text = f"<h3>Impact Analysis: {event['type']}</h3>"
        analysis_text += "<p>Changes in key economic indicators before and after the event:</p>"
        
        for indicator in indicators:
            if indicator in y_values_dict:
                values = y_values_dict[indicator]
                
                if None not in values[-3:] and None not in values[:3]:
                    before_avg = sum(values[:3]) / 3
                    after_avg = sum(values[-3:]) / 3
                    change = after_avg - before_avg
                    
                    if abs(change) < 0.01:
                        impact = "No significant impact"
                    else:
                        impact = f"<span style='color:{Config.COLORS['positive']};'>Increased</span>" if change > 0 else f"<span style='color:{Config.COLORS['negative']};'>Decreased</span>"
                        
                        # For unemployment and inflation, decrease is good
                        if indicator in ["Unemployment", "Inflation"]:
                            impact = f"<span style='color:{Config.COLORS['negative']};'>Increased</span>" if change > 0 else f"<span style='color:{Config.COLORS['positive']};'>Decreased</span>"
                    
                    analysis_text += f"<p><b>{indicator}:</b> {impact} by {abs(change):.2f} percentage points</p>"
        
        self.event_impact_analysis.setHtml(analysis_text)

    def show_person_details(self):
        """Show detailed information about selected person"""
        selected_rows = self.people_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(self, "No Selection", "Please select a person to view details.")
            return
            
        row = selected_rows[0].row()
        person_id = int(self.people_table.item(row, 0).text())
        
        # Find the person
        person = None
        for p in self.economy.persons:
            if p.id == person_id:
                person = p
                break
                
        if not person:
            QMessageBox.warning(self, "Person Not Found", "The selected person cannot be found.")
            return
            
        # Create and show person details dialog
        dialog = PersonDetailsDialog(person, self.economy, self)
        dialog.exec()

    def show_company_details(self):
        """Show detailed information about selected company"""
        selected_rows = self.companies_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(self, "No Selection", "Please select a company to view details.")
            return
            
        row = selected_rows[0].row()
        company_id = self.companies_table.item(row, 0).text()
        
        # Find the company
        company = None
        for c in self.economy.companies:
            if str(c.id)[:8] == company_id:
                company = c
                break
                
        if not company:
            QMessageBox.warning(self, "Company Not Found", "The selected company cannot be found.")
            return
            
        # Create and show company details dialog
        dialog = CompanyDetailsDialog(company, self.economy, self)
        dialog.exec()

    def show_population_stats(self):
        """Show detailed population statistics"""
        dialog = PopulationStatsDialog(self.economy, self)
        dialog.exec()

    def show_industry_stats(self):
        """Show detailed industry statistics"""
        dialog = IndustryStatsDialog(self.economy, self)
        dialog.exec()

    def apply_tax_changes(self):
        """Apply changes to tax policy settings"""
        # Update income tax brackets
        new_brackets = []
        for i, (label, slider, _) in enumerate(self.income_tax_widgets):
            bracket_text = label.text()
            rate = slider.value() / 100
            
            # Parse bracket bounds
            if "Over" in bracket_text:
                # Last bracket
                lower_bound = float(bracket_text.replace("Over £", "").replace(",", ""))
                upper_bound = float('inf')
            else:
                # Regular bracket
                bounds = bracket_text.replace("£", "").replace(",", "").split(" - ")
                lower_bound = float(bounds[0])
                upper_bound = float(bounds[1])
                
            new_brackets.append((lower_bound, upper_bound, rate))
            
        self.economy.government.update_tax_brackets(new_brackets)
        
        # Update other tax rates
        self.economy.government.corporation_tax_rate = self.corp_tax_slider.value() / 100
        self.economy.government.sales_tax_rate = self.sales_tax_slider.value() / 100
        self.economy.government.national_insurance_rate = self.ni_slider.value() / 100
        
        QMessageBox.information(self, "Tax Changes Applied", "Tax policy changes have been applied.")

    def apply_spending_changes(self):
        """Apply changes to government spending allocations"""
        new_allocation = {}
        total_allocation = 0
        
        for dept, (slider, _) in self.spending_widgets.items():
            rate = slider.value() / 1000  # Convert from slider value to percent of GDP
            new_allocation[dept] = rate
            total_allocation += rate
            
        # Check if allocations are reasonable
        if total_allocation < 0.15:
            QMessageBox.warning(self, "Low Spending", "Total government spending is very low (less than 15% of GDP). This may lead to economic issues.")
        elif total_allocation > 0.6:
            QMessageBox.warning(self, "High Spending", "Total government spending is very high (over 60% of GDP). This may lead to fiscal problems.")
            
        self.economy.government.update_spending_allocation(new_allocation)
        
        QMessageBox.information(self, "Spending Changes Applied", "Government spending changes have been applied.")

    def apply_monetary_policy(self):
        """Apply changes to monetary policy"""
        # Update interest rate
        new_interest_rate = self.interest_slider.value() / 1000
        self.economy.central_bank.set_interest_rate(new_interest_rate)
        
        # Update inflation target
        new_inflation_target = self.inflation_slider.value() / 1000
        self.economy.central_bank.inflation_target = new_inflation_target
        
        # Update reserve ratio
        new_reserve_ratio = self.reserve_slider.value() / 1000
        self.economy.central_bank.reserve_ratio = new_reserve_ratio
        
        QMessageBox.information(self, "Monetary Policy Applied", "Monetary policy changes have been applied.")

    def generate_policy_analysis(self):
        """Generate monetary policy analysis text"""
        # Basic analysis
        analysis = "<h3>Monetary Policy Analysis</h3>"
        
        # Check the relationship between inflation and interest rate
        inflation = self.economy.global_conditions["inflation"]
        interest_rate = self.economy.central_bank.interest_rate
        inflation_target = self.economy.central_bank.inflation_target
        
        # Interest rate vs inflation assessment
        interest_inflation_gap = interest_rate - inflation
        
        if inflation > inflation_target + 0.02:
            if interest_inflation_gap < -0.02:
                analysis += "<p>Interest rates appear <span style='color:red;'>too low</span> to control the current high inflation. Consider raising rates.</p>"
            else:
                analysis += "<p>Interest rates are positioned to help reduce inflation, but it may take time for effects to fully materialize.</p>"
        elif inflation < inflation_target - 0.02:
            if interest_inflation_gap > 0.02:
                analysis += "<p>Interest rates appear <span style='color:red;'>too high</span> given the low inflation environment. Consider reducing rates to stimulate growth.</p>"
            else:
                analysis += "<p>Interest rates are appropriate for the current low inflation environment.</p>"
        else:
            analysis += "<p>Interest rates are well-balanced with current inflation, which is near the target.</p>"
            
        # Economic growth impact
        economic_growth = self.economy.global_conditions["economic_growth"]
        if economic_growth < 0:
            if interest_rate > 0.04:
                analysis += "<p>High interest rates may be <span style='color:red;'>contributing to negative growth</span>. Consider easing monetary policy if inflation permits.</p>"
            else:
                analysis += "<p>Despite low interest rates, the economy is experiencing negative growth, suggesting structural issues beyond monetary policy.</p>"
        elif economic_growth < 0.01:
            if interest_rate > 0.06:
                analysis += "<p>Interest rates may be <span style='color:orange;'>constraining growth potential</span>. Consider a more accommodative stance if inflation is moderate.</p>"
        elif economic_growth > 0.04:
            if interest_rate < 0.03 and inflation > inflation_target:
                analysis += "<p>Economy showing strong growth with low rates. Consider <span style='color:green;'>normalizing monetary policy</span> to prevent overheating.</p>"
                
        # Money supply analysis
        if len(self.economy.central_bank.money_supply_history) > 12:
            annual_money_growth = (self.economy.central_bank.money_supply / self.economy.central_bank.money_supply_history[-13]) - 1
            
            if annual_money_growth > 0.15:
                analysis += f"<p>Money supply growing at <span style='color:red;'>{annual_money_growth*100:.1f}%</span> annually, which may fuel inflation if it continues.</p>"
            elif annual_money_growth < 0.01:
                analysis += f"<p>Slow money supply growth of <span style='color:orange;'>{annual_money_growth*100:.1f}%</span> may constrain economic expansion.</p>"
                
        # Interest rate trend
        if len(self.economy.central_bank.interest_rate_history) > 6:
            rate_6m_ago = self.economy.central_bank.interest_rate_history[-7]
            rate_change = interest_rate - rate_6m_ago
            
            if abs(rate_change) > 0.01:
                direction = "increased" if rate_change > 0 else "decreased"
                analysis += f"<p>Central bank has <span style='color:blue;'>{direction} rates by {abs(rate_change)*100:.2f}%</span> over the last 6 months.</p>"
                
        # Policy recommendation
        analysis += "<h4>Policy Recommendation:</h4>"
        
        if inflation > inflation_target + 0.02 and interest_rate < inflation:
            analysis += "<p><strong>Consider increasing interest rates</strong> to bring inflation back toward target.</p>"
        elif inflation < inflation_target - 0.01 and economic_growth < 0.02 and interest_rate > 0.03:
            analysis += "<p><strong>Consider decreasing interest rates</strong> to stimulate growth while inflation is below target.</p>"
        elif abs(inflation - inflation_target) < 0.01 and abs(economic_growth - 0.02) < 0.01:
            analysis += "<p><strong>Maintain current policy stance</strong> as both inflation and growth are near optimal levels.</p>"
        else:
            # Mixed signals
            if inflation > inflation_target and economic_growth < 0:
                analysis += "<p><strong>Challenging policy environment</strong> with stagflation signals. Consider focusing on structural reforms alongside careful monetary adjustments.</p>"
            elif inflation < inflation_target and economic_growth < 0:
                analysis += "<p><strong>Accommodative monetary policy recommended</strong> to address both low inflation and low growth.</p>"
                
        self.policy_analysis.setHtml(analysis)

    def generate_housing_analysis(self):
        """Generate housing market analysis text"""
        analysis = "<h3>Housing Market Analysis</h3>"
        
        # Price trend analysis
        if len(self.economy.housing_market.price_history) >= 13:
            annual_growth = (self.economy.housing_market.price_history[-1] / self.economy.housing_market.price_history[-13] - 1) * 100
            
            if annual_growth > 10:
                analysis += f"<p>Housing prices have <span style='color:red;'>increased dramatically</span> by {annual_growth:.1f}% over the past year, indicating potential overheating.</p>"
            elif annual_growth > 5:
                analysis += f"<p>Housing prices have <span style='color:orange;'>increased significantly</span> by {annual_growth:.1f}% over the past year.</p>"
            elif annual_growth > 2:
                analysis += f"<p>Housing prices have <span style='color:green;'>increased moderately</span> by {annual_growth:.1f}% over the past year.</p>"
            elif annual_growth >= 0:
                analysis += f"<p>Housing prices have <span style='color:blue;'>increased slightly</span> by {annual_growth:.1f}% over the past year.</p>"
            else:
                analysis += f"<p>Housing prices have <span style='color:purple;'>decreased</span> by {abs(annual_growth):.1f}% over the past year.</p>"
                
        # Affordability analysis
        try:
            price_to_income = self.economy.housing_market.avg_house_price / self.economy.avg_income
        except:
            price_to_income = 5.0
        if price_to_income > 10:
            analysis += f"<p>Housing affordability is <span style='color:red;'>extremely low</span> with price-to-income ratio of {price_to_income:.1f}x.</p>"
        elif price_to_income > 7:
            analysis += f"<p>Housing affordability is <span style='color:orange;'>very low</span> with price-to-income ratio of {price_to_income:.1f}x.</p>"
        elif price_to_income > 5:
            analysis += f"<p>Housing affordability is <span style='color:yellow;'>moderately low</span> with price-to-income ratio of {price_to_income:.1f}x.</p>"
        else:
            analysis += f"<p>Housing affordability is <span style='color:green;'>relatively good</span> with price-to-income ratio of {price_to_income:.1f}x.</p>"
            
        # Bubble risk analysis
        bubble_factor = self.economy.housing_market.bubble_factor
        
        if bubble_factor > 0.8:
            analysis += "<p>Housing bubble risk is <span style='color:red;'>extremely high</span>. Price correction is increasingly likely.</p>"
        elif bubble_factor > 0.6:
            analysis += "<p>Housing bubble risk is <span style='color:orange;'>high</span>. Market may be significantly overvalued.</p>"
        elif bubble_factor > 0.4:
            analysis += "<p>Housing bubble risk is <span style='color:yellow;'>moderate</span>. Some signs of overvaluation are present.</p>"
        else:
            analysis += "<p>Housing bubble risk is <span style='color:green;'>low</span>. Prices appear to be supported by fundamentals.</p>"
            
        # Interest rate impact
        interest_rate = self.economy.central_bank.interest_rate
        
        if interest_rate < 0.03:
            analysis += "<p>Low interest rates are <span style='color:blue;'>supporting housing demand</span> through affordable mortgage financing.</p>"
        elif interest_rate > 0.08:
            analysis += "<p>High interest rates are <span style='color:orange;'>constraining housing demand</span> through more expensive mortgage financing.</p>"
            
        # Rental market analysis
        rental_yield = self.economy.housing_market.rental_yield
        
        if rental_yield > 0.06:
            analysis += f"<p>Rental yields are <span style='color:green;'>high</span> at {rental_yield*100:.2f}%, suggesting good investment potential in rental properties.</p>"
        elif rental_yield < 0.03:
            analysis += f"<p>Rental yields are <span style='color:orange;'>low</span> at {rental_yield*100:.2f}%, suggesting housing may be overvalued relative to rental income.</p>"
            
        # Regional analysis
        urban_premium = self.economy.housing_market.regional_prices['Urban'] / self.economy.housing_market.regional_prices['Rural'] - 1
        
        if urban_premium > 2:
            analysis += f"<p>Urban housing commands a <span style='color:purple;'>very large premium</span> of {urban_premium*100:.1f}% over rural areas.</p>"
            
        # Market outlook
        analysis += "<h4>Market Outlook:</h4>"
        
        if bubble_factor > 0.7 and price_to_income > 8:
            analysis += "<p><strong>Negative outlook</strong>. High risk of price correction or crash.</p>"
        elif (bubble_factor > 0.5 or price_to_income > 7) and interest_rate > 0.06:
            analysis += "<p><strong>Cautious outlook</strong>. Market may face headwinds from high interest rates and valuation concerns.</p>"
        elif bubble_factor < 0.3 and interest_rate < 0.04 and self.economy.global_conditions["economic_growth"] > 0.02:
            analysis += "<p><strong>Positive outlook</strong>. Supportive economic conditions and reasonable valuations suggest continued steady growth.</p>"
        else:
            analysis += "<p><strong>Neutral outlook</strong>. Mixed signals suggest prices may continue current trajectory with moderate fluctuations.</p>"
            
        self.housing_analysis.setHtml(analysis)

class PersonDetailsDialog(QDialog):
    """Dialog showing detailed information about a person"""
    def __init__(self, person, economy, parent=None):
        super().__init__(parent)
        self.person = person
        self.economy = economy
        
        self.setWindowTitle(f"Person Details: {person.name}")
        self.setMinimumSize(800, 600)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Header with basic info
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        
        name_label = QLabel(self.person.name)
        name_label.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {Config.COLORS['primary']};")
        header_layout.addWidget(name_label)
        
        header_layout.addStretch()
        
        id_label = QLabel(f"ID: {self.person.id}")
        id_label.setStyleSheet(f"color: {Config.COLORS['text_secondary']};")
        header_layout.addWidget(id_label)
        
        layout.addWidget(header_widget)
        
        # Main content area
        content_widget = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Personal information
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Basic demographics group
        demographics_group = QGroupBox("Demographics")
        demographics_layout = QFormLayout(demographics_group)
        
        demographics_layout.addRow("Age:", QLabel(f"{self.person.age} years"))
        demographics_layout.addRow("Gender:", QLabel(self.person.gender))
        demographics_layout.addRow("Education:", QLabel(self.person.education_level))
        demographics_layout.addRow("Political Views:", QLabel(self.person.political_views))
        demographics_layout.addRow("Housing Status:", QLabel(self.person.housing_status))
        
        left_layout.addWidget(demographics_group)
        
        # Personality traits group
        traits_group = QGroupBox("Personality & Skills")
        traits_layout = QFormLayout(traits_group)
        
        traits_layout.addRow("Risk Tolerance:", QLabel(f"{self.person.risk_tolerance*100:.1f}%"))
        traits_layout.addRow("Entrepreneurship:", QLabel(f"{self.person.entrepreneurship*100:.1f}%"))
        traits_layout.addRow("Work Ethic:", QLabel(f"{self.person.work_ethic*100:.1f}%"))
        traits_layout.addRow("Innovation:", QLabel(f"{self.person.innovation*100:.1f}%"))
        traits_layout.addRow("Financial Literacy:", QLabel(f"{self.person.financial_literacy*100:.1f}%"))
        
        for skill, value in self.person.skills.items():
            traits_layout.addRow(f"{skill.capitalize()} Skill:", QLabel(f"{value*100:.1f}%"))
            
        left_layout.addWidget(traits_group)
        
        # Employment information group
        employment_group = QGroupBox("Employment")
        employment_layout = QFormLayout(employment_group)
        
        employment_layout.addRow("Status:", QLabel(self.person.employment_status))
        employment_layout.addRow("Job Title:", QLabel(self.person.job_title if self.person.job_title else "N/A"))
        employment_layout.addRow("Employer:", QLabel(self.person.employer.name if self.person.employer else "N/A"))
        employment_layout.addRow("Years Experience:", QLabel(f"{self.person.years_experience} years"))
        employment_layout.addRow("Job Satisfaction:", QLabel(f"{self.person.job_satisfaction*100:.1f}%"))
        employment_layout.addRow("Productivity:", QLabel(f"{self.person.productivity*100:.1f}%"))
        
        left_layout.addWidget(employment_group)
        
        # Right panel - Economic information and charts
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Economic information group
        economic_group = QGroupBox("Economic Status")
        economic_layout = QFormLayout(economic_group)
        
        economic_layout.addRow("Monthly Income:", QLabel(format_currency(self.person.monthly_income)))
        economic_layout.addRow("Annual Income:", QLabel(format_currency(self.person.monthly_income * 12)))
        economic_layout.addRow("Tax Paid (Monthly):", QLabel(format_currency(self.person.tax_paid)))
        economic_layout.addRow("Disposable Income:", QLabel(format_currency(self.person.disposable_income)))
        economic_layout.addRow("Savings:", QLabel(format_currency(self.person.savings)))
        economic_layout.addRow("Savings Rate:", QLabel(f"{self.person.savings_rate*100:.1f}%"))
        economic_layout.addRow("Net Worth:", QLabel(format_currency(self.person.net_worth)))
        
        if self.person.is_business_owner:
            economic_layout.addRow("Business Value:", QLabel(format_currency(self.person.business.value if self.person.business else 0)))
            
        economic_layout.addRow("Housing Value:", QLabel(format_currency(self.person.housing_value)))
        economic_layout.addRow("Housing Payment:", QLabel(format_currency(self.person.housing_payment)))
        economic_layout.addRow("Investment Value:", QLabel(format_currency(self.person.investment_value)))
        
        right_layout.addWidget(economic_group)
        
        # Charts group
        charts_group = QGroupBox("Financial History")
        charts_layout = QVBoxLayout(charts_group)
        
        # Income history chart
        income_chart_label = QLabel("Income History")
        income_chart_label.setStyleSheet("font-weight: bold;")
        charts_layout.addWidget(income_chart_label)
        
        # Create income chart
        income_chart = self.create_chart(self.person.income_history, "Monthly Income", Config.COLORS["positive"])
        charts_layout.addWidget(income_chart)
        
        # Net worth history chart
        networth_chart_label = QLabel("Net Worth History")
        networth_chart_label.setStyleSheet("font-weight: bold;")
        charts_layout.addWidget(networth_chart_label)
        
        # Create net worth chart
        networth_chart = self.create_chart(self.person.net_worth_history, "Net Worth", Config.COLORS["primary"])
        charts_layout.addWidget(networth_chart)
        
        right_layout.addWidget(charts_group)
        
        # Life events group
        events_group = QGroupBox("Life Events")
        events_layout = QVBoxLayout(events_group)
        
        events_table = QTableWidget()
        events_table.setColumnCount(2)
        events_table.setHorizontalHeaderLabels(["Date", "Event"])
        events_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        events_table.setRowCount(len(self.person.life_events))
        
        for i, (event, date) in enumerate(self.person.life_events):
            date_item = QTableWidgetItem(date.strftime("%b %Y") if hasattr(date, "strftime") else str(date))
            event_item = QTableWidgetItem(event)
            
            events_table.setItem(i, 0, date_item)
            events_table.setItem(i, 1, event_item)
            
        events_layout.addWidget(events_table)
        
        right_layout.addWidget(events_group)
        
        # Add panels to splitter
        content_widget.addWidget(left_widget)
        content_widget.addWidget(right_widget)
        
        # Set initial sizes
        content_widget.setSizes([400, 400])
        
        layout.addWidget(content_widget)
        
        # Bottom buttons
        buttons_layout = QHBoxLayout()
        
        if self.person.is_business_owner and self.person.business:
            view_business_button = ModernPushButton("View Business", True)
            view_business_button.clicked.connect(self.view_business)
            buttons_layout.addWidget(view_business_button)
            
        close_button = ModernPushButton("Close", False)
        close_button.clicked.connect(self.accept)
        buttons_layout.addWidget(close_button)
        
        layout.addLayout(buttons_layout)
        
    def create_chart(self, data, title, color):
        """Create a chart for the given data"""
        fig = Figure(figsize=(5, 3), dpi=100, facecolor=Config.COLORS["background"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(Config.COLORS["background"])
        
        x_values = list(range(len(data)))
        ax.plot(x_values, data, marker='o', color=color, linewidth=2)
        
        ax.set_title(title, color=Config.COLORS["text_primary"])
        ax.tick_params(axis='x', colors=Config.COLORS["text_secondary"])
        ax.tick_params(axis='y', colors=Config.COLORS["text_secondary"])
        ax.grid(True, linestyle='--', alpha=0.3, color=Config.COLORS["chart_grid"])
        
        canvas = FigureCanvas(fig)
        canvas.setFixedHeight(200)
        
        return canvas
        
    def view_business(self):
        """View the person's business"""
        if self.person.is_business_owner and self.person.business:
            dialog = CompanyDetailsDialog(self.person.business, self.economy, self)
            dialog.exec()

class CompanyDetailsDialog(QDialog):
    """Dialog showing detailed information about a company"""
    def __init__(self, company, economy, parent=None):
        super().__init__(parent)
        self.company = company
        self.economy = economy
        
        self.setWindowTitle(f"Company Details: {company.name}")
        self.setMinimumSize(800, 600)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Header with basic info
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        
        name_label = QLabel(self.company.name)
        name_label.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {Config.COLORS['primary']};")
        header_layout.addWidget(name_label)
        
        header_layout.addStretch()
        
        id_label = QLabel(f"ID: {self.company.id}")
        id_label.setStyleSheet(f"color: {Config.COLORS['text_secondary']};")
        header_layout.addWidget(id_label)
        
        layout.addWidget(header_widget)
        
        # Main content area
        content_widget = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Company information
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Basic information group
        info_group = QGroupBox("General Information")
        info_layout = QFormLayout(info_group)
        
        info_layout.addRow("Sector:", QLabel(self.company.sector))
        info_layout.addRow("Business Type:", QLabel(self.company.business_type))
        info_layout.addRow("Age:", QLabel(f"{self.company.age} months"))
        info_layout.addRow("Public Company:", QLabel("Yes" if self.company.is_public else "No"))
        
        if hasattr(self.company, 'founding_date'):
            info_layout.addRow("Founded:", QLabel(self.company.founding_date.strftime("%b %Y")))
            
        # Find owner
        owner = None
        for person in self.economy.persons:
            if person.id == self.company.owner_id:
                owner = person
                break
                
        info_layout.addRow("Owner:", QLabel(owner.name if owner else "Unknown"))
        
        left_layout.addWidget(info_group)
        
        # Business metrics group
        metrics_group = QGroupBox("Business Metrics")
        metrics_layout = QFormLayout(metrics_group)
        
        metrics_layout.addRow("Employees:", QLabel(str(len(self.company.employees))))
        metrics_layout.addRow("Customer Satisfaction:", QLabel(f"{self.company.customer_satisfaction*100:.1f}%"))
        metrics_layout.addRow("Innovation Level:", QLabel(f"{self.company.innovation_level*100:.1f}%"))
        metrics_layout.addRow("Efficiency:", QLabel(f"{self.company.efficiency*100:.1f}%"))
        metrics_layout.addRow("Technology Level:", QLabel(f"{self.company.technology_level*100:.1f}%"))
        
        if hasattr(self.company, 'market_share'):
            metrics_layout.addRow("Market Share:", QLabel(f"{self.company.market_share*100:.1f}%"))
            
        if hasattr(self.company, 'growth_rate'):
            growth_color = "green" if self.company.growth_rate > 0 else "red"
            metrics_layout.addRow("Growth Rate:", QLabel(f"<span style='color:{growth_color};'>{self.company.growth_rate*100:.1f}%</span>"))
        
        left_layout.addWidget(metrics_group)
        
        # Stock information (if public)
        if self.company.is_public:
            stock_group = QGroupBox("Stock Information")
            stock_layout = QFormLayout(stock_group)
            
            if hasattr(self.company, 'share_price'):
                stock_layout.addRow("Share Price:", QLabel(format_currency(self.company.share_price)))
                
            stock_layout.addRow("Shares Outstanding:", QLabel(f"{self.company.shares_outstanding:,}"))
            stock_layout.addRow("Market Cap:", QLabel(format_currency(self.company.value)))
            
            # Calculate P/E ratio
            if hasattr(self.company, 'profit') and self.company.profit > 0:
                pe_ratio = (self.company.share_price * self.company.shares_outstanding) / (self.company.profit * 12)
                stock_layout.addRow("P/E Ratio:", QLabel(f"{pe_ratio:.1f}x"))
            
            left_layout.addWidget(stock_group)
            
        # Right panel - Financial information and charts
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Financial information group
        financial_group = QGroupBox("Financial Information")
        financial_layout = QFormLayout(financial_group)
        
        financial_layout.addRow("Monthly Revenue:", QLabel(format_currency(self.company.revenue)))
        financial_layout.addRow("Annual Revenue:", QLabel(format_currency(self.company.revenue * 12)))
        financial_layout.addRow("Monthly Profit:", QLabel(format_currency(self.company.profit)))
        financial_layout.addRow("Annual Profit:", QLabel(format_currency(self.company.profit * 12)))
        
        if hasattr(self.company, 'costs'):
            financial_layout.addRow("Monthly Costs:", QLabel(format_currency(self.company.costs)))
            
        if hasattr(self.company, 'profitability'):
            profit_color = "green" if self.company.profitability > 0 else "red"
            financial_layout.addRow("Profit Margin:", QLabel(f"<span style='color:{profit_color};'>{self.company.profitability*100:.1f}%</span>"))
            
        financial_layout.addRow("Company Value:", QLabel(format_currency(self.company.value)))
        
        if hasattr(self.company, 'capital'):
            financial_layout.addRow("Capital:", QLabel(format_currency(self.company.capital)))
            
        if hasattr(self.company, 'r_and_d_investment'):
            financial_layout.addRow("R&D Investment:", QLabel(format_currency(self.company.r_and_d_investment)))
            
        financial_layout.addRow("Tax Paid:", QLabel(format_currency(self.company.tax_paid) if hasattr(self.company, 'tax_paid') else "N/A"))
        
        right_layout.addWidget(financial_group)
        
        # Charts group
        charts_group = QGroupBox("Financial History")
        charts_layout = QVBoxLayout(charts_group)
        
        # Revenue history chart
        if hasattr(self.company, 'revenue_history') and len(self.company.revenue_history) > 0:
            revenue_chart_label = QLabel("Revenue History")
            revenue_chart_label.setStyleSheet("font-weight: bold;")
            charts_layout.addWidget(revenue_chart_label)
            
            revenue_chart = self.create_chart(self.company.revenue_history, "Monthly Revenue", Config.COLORS["positive"])
            charts_layout.addWidget(revenue_chart)
            
        # Profit history chart
        if hasattr(self.company, 'profit_history') and len(self.company.profit_history) > 0:
            profit_chart_label = QLabel("Profit History")
            profit_chart_label.setStyleSheet("font-weight: bold;")
            charts_layout.addWidget(profit_chart_label)
            
            profit_chart = self.create_chart(self.company.profit_history, "Monthly Profit", Config.COLORS["primary"])
            charts_layout.addWidget(profit_chart)
            
        # Stock price chart (if public)
        if self.company.is_public and hasattr(self.company, 'share_price_history') and len(self.company.share_price_history) > 0:
            stock_chart_label = QLabel("Share Price History")
            stock_chart_label.setStyleSheet("font-weight: bold;")
            charts_layout.addWidget(stock_chart_label)
            
            stock_chart = self.create_chart(self.company.share_price_history, "Share Price", Config.COLORS["warning"])
            charts_layout.addWidget(stock_chart)
            
        right_layout.addWidget(charts_group)
        
        # Add panels to splitter
        content_widget.addWidget(left_widget)
        content_widget.addWidget(right_widget)
        
        # Set initial sizes
        content_widget.setSizes([400, 400])
        
        layout.addWidget(content_widget)
        
        # Employees section
        employees_group = QGroupBox("Employees")
        employees_layout = QVBoxLayout(employees_group)
        
        employees_table = QTableWidget()
        employees_table.setColumnCount(7)
        employees_table.setHorizontalHeaderLabels([
            "ID", "Name", "Job Title", "Age", "Education",
            "Productivity", "Satisfaction"
        ])
        employees_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        employees_table.setRowCount(len(self.company.employees))
        
        for i, employee in enumerate(self.company.employees):
            # ID
            id_item = QTableWidgetItem(str(employee.id))
            employees_table.setItem(i, 0, id_item)
            
            # Name
            name_item = QTableWidgetItem(employee.name)
            employees_table.setItem(i, 1, name_item)
            
            # Job Title
            job_item = QTableWidgetItem(employee.job_title if employee.job_title else "Employee")
            employees_table.setItem(i, 2, job_item)
            
            # Age
            age_item = QTableWidgetItem(str(employee.age))
            employees_table.setItem(i, 3, age_item)
            
            # Education
            education_item = QTableWidgetItem(employee.education_level)
            employees_table.setItem(i, 4, education_item)
            
            # Productivity
            productivity_item = QTableWidgetItem(f"{employee.productivity*100:.1f}%")
            employees_table.setItem(i, 5, productivity_item)
            
            # Satisfaction
            satisfaction_item = QTableWidgetItem(f"{employee.job_satisfaction*100:.1f}%")
            employees_table.setItem(i, 6, satisfaction_item)
            
        employees_layout.addWidget(employees_table)
        
        layout.addWidget(employees_group)
        
        # Bottom buttons
        buttons_layout = QHBoxLayout()
        
        view_owner_button = ModernPushButton("View Owner", True)
        view_owner_button.clicked.connect(self.view_owner)
        buttons_layout.addWidget(view_owner_button)
        
        close_button = ModernPushButton("Close", False)
        close_button.clicked.connect(self.accept)
        buttons_layout.addWidget(close_button)
        
        layout.addLayout(buttons_layout)
        
    def create_chart(self, data, title, color):
        """Create a chart for the given data"""
        fig = Figure(figsize=(5, 3), dpi=100, facecolor=Config.COLORS["background"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(Config.COLORS["background"])
        
        x_values = list(range(len(data)))
        ax.plot(x_values, data, marker='o', color=color, linewidth=2)
        
        ax.set_title(title, color=Config.COLORS["text_primary"])
        ax.tick_params(axis='x', colors=Config.COLORS["text_secondary"])
        ax.tick_params(axis='y', colors=Config.COLORS["text_secondary"])
        ax.grid(True, linestyle='--', alpha=0.3, color=Config.COLORS["chart_grid"])
        
        canvas = FigureCanvas(fig)
        canvas.setFixedHeight(200)
        
        return canvas
        
    def view_owner(self):
        """View the company owner"""
        owner = None
        for person in self.economy.persons:
            if person.id == self.company.owner_id:
                owner = person
                break
                
        if owner:
            dialog = PersonDetailsDialog(owner, self.economy, self)
            dialog.exec()
        else:
            QMessageBox.information(self, "Owner Not Found", "The owner of this company cannot be found.")

class PopulationStatsDialog(QDialog):
    """Dialog showing detailed population statistics"""
    def __init__(self, economy, parent=None):
        super().__init__(parent)
        self.economy = economy
        
        self.setWindowTitle("Population Statistics")
        self.setMinimumSize(900, 700)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Population Statistics")
        title_label.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {Config.COLORS['primary']};")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Main tabs
        tabs = QTabWidget()
        
        # Demographics tab
        demographics_tab = QWidget()
        demographics_layout = QVBoxLayout(demographics_tab)
        
        # Create demographics charts
        demographics_layout.addWidget(self.create_demographics_charts())
        
        tabs.addTab(demographics_tab, "Demographics")
        
        # Income & Wealth tab
        income_tab = QWidget()
        income_layout = QVBoxLayout(income_tab)
        
        # Create income charts
        income_layout.addWidget(self.create_income_charts())
        
        tabs.addTab(income_tab, "Income & Wealth")
        
        # Employment tab
        employment_tab = QWidget()
        employment_layout = QVBoxLayout(employment_tab)
        
        # Create employment charts
        employment_layout.addWidget(self.create_employment_charts())
        
        tabs.addTab(employment_tab, "Employment")
        
        # Education tab
        education_tab = QWidget()
        education_layout = QVBoxLayout(education_tab)
        
        # Create education charts
        education_layout.addWidget(self.create_education_charts())
        
        tabs.addTab(education_tab, "Education")
        
        # Housing tab
        housing_tab = QWidget()
        housing_layout = QVBoxLayout(housing_tab)
        
        # Create housing charts
        housing_layout.addWidget(self.create_housing_charts())
        
        tabs.addTab(housing_tab, "Housing")
        
        layout.addWidget(tabs)
        
        # Summary statistics
        summary_group = QGroupBox("Summary Statistics")
        summary_layout = QGridLayout(summary_group)
        
        # Calculate summary stats
        population = len(self.economy.persons)
        avg_age = sum(p.age for p in self.economy.persons) / population if population > 0 else 0
        median_income = sorted([p.monthly_income for p in self.economy.persons])[population // 2] * 12 if population > 0 else 0
        avg_income = sum(p.monthly_income for p in self.economy.persons) / population * 12 if population > 0 else 0
        employed = sum(1 for p in self.economy.persons if p.employment_status != "Unemployed")
        employment_rate = employed / population * 100 if population > 0 else 0
        business_owners = sum(1 for p in self.economy.persons if p.is_business_owner)
        homeowners = sum(1 for p in self.economy.persons if p.housing_status == "Owner")
        homeownership_rate = homeowners / population * 100 if population > 0 else 0
        
        # Add summary stats to grid
        summary_layout.addWidget(QLabel("Population:"), 0, 0)
        summary_layout.addWidget(QLabel(f"{population:,}"), 0, 1)
        
        summary_layout.addWidget(QLabel("Average Age:"), 0, 2)
        summary_layout.addWidget(QLabel(f"{avg_age:.1f} years"), 0, 3)
        
        summary_layout.addWidget(QLabel("Median Income:"), 1, 0)
        summary_layout.addWidget(QLabel(format_currency(median_income)), 1, 1)
        
        summary_layout.addWidget(QLabel("Average Income:"), 1, 2)
        summary_layout.addWidget(QLabel(format_currency(avg_income)), 1, 3)
        
        summary_layout.addWidget(QLabel("Employment Rate:"), 2, 0)
        summary_layout.addWidget(QLabel(f"{employment_rate:.1f}%"), 2, 1)
        
        summary_layout.addWidget(QLabel("Business Owners:"), 2, 2)
        summary_layout.addWidget(QLabel(f"{business_owners:,} ({business_owners/population*100:.1f}%)"), 2, 3)
        
        summary_layout.addWidget(QLabel("Homeownership Rate:"), 3, 0)
        summary_layout.addWidget(QLabel(f"{homeownership_rate:.1f}%"), 3, 1)
        
        layout.addWidget(summary_group)
        
        # Close button
        close_button = ModernPushButton("Close", False)
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
        
    def create_demographics_charts(self):
        """Create demographics charts"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Age distribution
        age_ranges = [
            "18-24", "25-34", "35-44", "45-54", 
            "55-64", "65-74", "75-84", "85+"
        ]
        
        age_counts = [0] * len(age_ranges)
        
        for person in self.economy.persons:
            if person.age < 25:
                age_counts[0] += 1
            elif person.age < 35:
                age_counts[1] += 1
            elif person.age < 45:
                age_counts[2] += 1
            elif person.age < 55:
                age_counts[3] += 1
            elif person.age < 65:
                age_counts[4] += 1
            elif person.age < 75:
                age_counts[5] += 1
            elif person.age < 85:
                age_counts[6] += 1
            else:
                age_counts[7] += 1
                
        # Normalize to percentages
        population = len(self.economy.persons)
        age_percentages = [count / population * 100 for count in age_counts] if population > 0 else [0] * len(age_ranges)
        
        age_chart = self.create_bar_chart(age_ranges, age_percentages, "Age Distribution", "Age Group", "Percentage (%)")
        layout.addWidget(age_chart, 0, 0)
        
        # Gender distribution
        gender_counts = {"Male": 0, "Female": 0, "Non-binary": 0}
        
        for person in self.economy.persons:
            gender_counts[person.gender] = gender_counts.get(person.gender, 0) + 1
            
        # Normalize to percentages
        gender_percentages = [count / population * 100 for count in gender_counts.values()] if population > 0 else [0] * len(gender_counts)
        
        gender_chart = self.create_bar_chart(list(gender_counts.keys()), gender_percentages, "Gender Distribution", "Gender", "Percentage (%)")
        layout.addWidget(gender_chart, 0, 1)
        
        # Political views distribution
        politics_counts = {
            "Far Left": 0, "Left": 0, "Center": 0, "Right": 0, "Far Right": 0
        }
        
        for person in self.economy.persons:
            politics_counts[person.political_views] = politics_counts.get(person.political_views, 0) + 1
            
        # Normalize to percentages
        politics_percentages = [count / population * 100 for count in politics_counts.values()] if population > 0 else [0] * len(politics_counts)
        
        politics_chart = self.create_bar_chart(list(politics_counts.keys()), politics_percentages, "Political Views Distribution", "Political View", "Percentage (%)")
        layout.addWidget(politics_chart, 1, 0)
        
        # Personality traits distribution
        traits = {
            "Risk Tolerance": 0,
            "Entrepreneurship": 0,
            "Work Ethic": 0,
            "Innovation": 0,
            "Financial Literacy": 0
        }
        
        for person in self.economy.persons:
            traits["Risk Tolerance"] += person.risk_tolerance
            traits["Entrepreneurship"] += person.entrepreneurship
            traits["Work Ethic"] += person.work_ethic
            traits["Innovation"] += person.innovation
            traits["Financial Literacy"] += person.financial_literacy
            
        # Calculate averages
        trait_averages = [value / population * 100 for value in traits.values()] if population > 0 else [0] * len(traits)
        
        traits_chart = self.create_bar_chart(list(traits.keys()), trait_averages, "Average Personality Traits", "Trait", "Average Value (%)")
        layout.addWidget(traits_chart, 1, 1)
        
        return widget
        
    def create_income_charts(self):
        """Create income and wealth charts"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Income distribution
        income_ranges = [
            "0-20k", "20k-40k", "40k-60k", "60k-80k", 
            "80k-100k", "100k-150k", "150k-200k", "200k+"
        ]
        
        income_counts = [0] * len(income_ranges)
        
        for person in self.economy.persons:
            annual_income = person.monthly_income * 12
            if annual_income < 20000:
                income_counts[0] += 1
            elif annual_income < 40000:
                income_counts[1] += 1
            elif annual_income < 60000:
                income_counts[2] += 1
            elif annual_income < 80000:
                income_counts[3] += 1
            elif annual_income < 100000:
                income_counts[4] += 1
            elif annual_income < 150000:
                income_counts[5] += 1
            elif annual_income < 200000:
                income_counts[6] += 1
            else:
                income_counts[7] += 1
                
        # Normalize to percentages
        population = len(self.economy.persons)
        income_percentages = [count / population * 100 for count in income_counts] if population > 0 else [0] * len(income_ranges)
        
        income_chart = self.create_bar_chart(income_ranges, income_percentages, "Income Distribution", "Annual Income Range", "Percentage (%)")
        layout.addWidget(income_chart, 0, 0)
        
        # Net worth distribution
        worth_ranges = [
            "0-50k", "50k-100k", "100k-250k", "250k-500k", 
            "500k-1M", "1M-2M", "2M-5M", "5M+"
        ]
        
        worth_counts = [0] * len(worth_ranges)
        
        for person in self.economy.persons:
            if person.net_worth < 50000:
                worth_counts[0] += 1
            elif person.net_worth < 100000:
                worth_counts[1] += 1
            elif person.net_worth < 250000:
                worth_counts[2] += 1
            elif person.net_worth < 500000:
                worth_counts[3] += 1
            elif person.net_worth < 1000000:
                worth_counts[4] += 1
            elif person.net_worth < 2000000:
                worth_counts[5] += 1
            elif person.net_worth < 5000000:
                worth_counts[6] += 1
            else:
                worth_counts[7] += 1
                
        # Normalize to percentages
        worth_percentages = [count / population * 100 for count in worth_counts] if population > 0 else [0] * len(worth_ranges)
        
        worth_chart = self.create_bar_chart(worth_ranges, worth_percentages, "Net Worth Distribution", "Net Worth Range", "Percentage (%)")
        layout.addWidget(worth_chart, 0, 1)
        
        # Savings rate distribution
        savings_ranges = [
            "0-5%", "5-10%", "10-15%", "15-20%", 
            "20-25%", "25-30%", "30%+"
        ]
        
        savings_counts = [0] * len(savings_ranges)
        
        for person in self.economy.persons:
            savings_rate = person.savings_rate * 100
            if savings_rate < 5:
                savings_counts[0] += 1
            elif savings_rate < 10:
                savings_counts[1] += 1
            elif savings_rate < 15:
                savings_counts[2] += 1
            elif savings_rate < 20:
                savings_counts[3] += 1
            elif savings_rate < 25:
                savings_counts[4] += 1
            elif savings_rate < 30:
                savings_counts[5] += 1
            else:
                savings_counts[6] += 1
                
        # Normalize to percentages
        savings_percentages = [count / population * 100 for count in savings_counts] if population > 0 else [0] * len(savings_ranges)
        
        savings_chart = self.create_bar_chart(savings_ranges, savings_percentages, "Savings Rate Distribution", "Savings Rate", "Percentage (%)")
        layout.addWidget(savings_chart, 1, 0)
        
        # Income by education level
        education_levels = [
            "No Formal Education", "Primary", "Secondary", 
            "College", "Bachelor's", "Master's", "Doctorate"
        ]
        
        education_incomes = {level: [] for level in education_levels}
        
        for person in self.economy.persons:
            if person.education_level in education_incomes:
                education_incomes[person.education_level].append(person.monthly_income * 12)
                
        # Calculate average income by education level
        education_avg_incomes = []
        for level in education_levels:
            incomes = education_incomes[level]
            avg = sum(incomes) / len(incomes) if incomes else 0
            education_avg_incomes.append(avg)
            
        education_income_chart = self.create_bar_chart(education_levels, education_avg_incomes, "Average Income by Education", "Education Level", "Average Annual Income (£)")
        layout.addWidget(education_income_chart, 1, 1)
        
        return widget
        
    def create_employment_charts(self):
        """Create employment charts"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Employment status distribution
        status_counts = {"Employed": 0, "Self-employed": 0, "Unemployed": 0}
        
        for person in self.economy.persons:
            if person.is_business_owner:
                status_counts["Self-employed"] += 1
            elif person.employment_status == "Employed":
                status_counts["Employed"] += 1
            else:
                status_counts["Unemployed"] += 1
                
        # Normalize to percentages
        population = len(self.economy.persons)
        status_percentages = [count / population * 100 for count in status_counts.values()] if population > 0 else [0] * len(status_counts)
        
        status_chart = self.create_bar_chart(list(status_counts.keys()), status_percentages, "Employment Status Distribution", "Status", "Percentage (%)")
        layout.addWidget(status_chart, 0, 0)
        
        # Sector distribution for employed people
        sector_counts = {}
        
        for person in self.economy.persons:
            if person.employer:
                sector = person.employer.sector
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
                
        # Sort sectors by count
        sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
        sectors = [s[0] for s in sorted_sectors]
        sector_counts = [s[1] for s in sorted_sectors]
        
        # Normalize to percentages of employed population
        employed_count = sum(sector_counts)
        sector_percentages = [count / employed_count * 100 for count in sector_counts] if employed_count > 0 else [0] * len(sectors)
        
        sector_chart = self.create_bar_chart(sectors, sector_percentages, "Employment by Sector", "Sector", "Percentage (%)")
        layout.addWidget(sector_chart, 0, 1)
        
        # Job satisfaction distribution
        satisfaction_ranges = [
            "0-20%", "20-40%", "40-60%", "60-80%", "80-100%"
        ]
        
        satisfaction_counts = [0] * len(satisfaction_ranges)
        
        for person in self.economy.persons:
            if person.employment_status != "Unemployed":
                satisfaction = person.job_satisfaction * 100
                idx = min(int(satisfaction / 20), 4)
                satisfaction_counts[idx] += 1
                
        # Normalize to percentages of employed population
        satisfaction_percentages = [count / employed_count * 100 for count in satisfaction_counts] if employed_count > 0 else [0] * len(satisfaction_ranges)
        
        satisfaction_chart = self.create_bar_chart(satisfaction_ranges, satisfaction_percentages, "Job Satisfaction Distribution", "Satisfaction Level", "Percentage (%)")
        layout.addWidget(satisfaction_chart, 1, 0)
        
        # Productivity distribution
        productivity_ranges = [
            "0-20%", "20-40%", "40-60%", "60-80%", "80-100%"
        ]
        
        productivity_counts = [0] * len(productivity_ranges)
        
        for person in self.economy.persons:
            if person.employment_status != "Unemployed":
                productivity = person.productivity * 100
                idx = min(int(productivity / 20), 4)
                productivity_counts[idx] += 1
                
        # Normalize to percentages of employed population
        productivity_percentages = [count / employed_count * 100 for count in productivity_counts] if employed_count > 0 else [0] * len(productivity_ranges)
        
        productivity_chart = self.create_bar_chart(productivity_ranges, productivity_percentages, "Productivity Distribution", "Productivity Level", "Percentage (%)")
        layout.addWidget(productivity_chart, 1, 1)
        
        return widget
        
    def create_education_charts(self):
        """Create education charts"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Education level distribution
        education_counts = {
            "No Formal Education": 0,
            "Primary": 0,
            "Secondary": 0,
            "College": 0,
            "Bachelor's": 0,
            "Master's": 0,
            "Doctorate": 0
        }
        
        for person in self.economy.persons:
            education_counts[person.education_level] = education_counts.get(person.education_level, 0) + 1
            
        # Normalize to percentages
        population = len(self.economy.persons)
        education_percentages = [count / population * 100 for count in education_counts.values()] if population > 0 else [0] * len(education_counts)
        
        education_chart = self.create_bar_chart(list(education_counts.keys()), education_percentages, "Education Level Distribution", "Education Level", "Percentage (%)")
        layout.addWidget(education_chart, 0, 0)
        
        # Education level by age group
        age_groups = ["18-34", "35-54", "55+"]
        
        education_by_age = {
            age_group: {level: 0 for level in education_counts.keys()}
            for age_group in age_groups
        }
        
        age_group_counts = {group: 0 for group in age_groups}
        
        for person in self.economy.persons:
            if person.age < 35:
                age_group = "18-34"
            elif person.age < 55:
                age_group = "35-54"
            else:
                age_group = "55+"
                
            education_by_age[age_group][person.education_level] += 1
            age_group_counts[age_group] += 1
            
        # Select a subset of education levels for clarity
        selected_levels = ["No Formal Education", "Secondary", "Bachelor's", "Master's", "Doctorate"]
        
        # Calculate percentages
        edu_age_data = {}
        for level in selected_levels:
            edu_age_data[level] = []
            for group in age_groups:
                percentage = education_by_age[group][level] / age_group_counts[group] * 100 if age_group_counts[group] > 0 else 0
                edu_age_data[level].append(percentage)
                
        education_age_chart = self.create_multi_bar_chart(age_groups, edu_age_data, "Education Level by Age Group", "Age Group", "Percentage (%)")
        layout.addWidget(education_age_chart, 0, 1)
        
        # Skills distribution
        skill_types = ["technical", "managerial", "creative", "analytical", "communication", "physical", "financial"]
        
        skill_averages = {skill: 0 for skill in skill_types}
        
        for person in self.economy.persons:
            for skill in skill_types:
                skill_averages[skill] += person.skills.get(skill, 0)
                
        # Calculate average
        skill_averages = [value / population * 100 for value in skill_averages.values()] if population > 0 else [0] * len(skill_types)
        
        skills_chart = self.create_bar_chart(skill_types, skill_averages, "Average Skill Levels", "Skill Type", "Average Level (%)")
        layout.addWidget(skills_chart, 1, 0)
        
        # Education impact on income and wealth
        metrics = ["Income", "Net Worth"]
        
        education_impact = {
            "No Formal Education": [0, 0],
            "Secondary": [0, 0],
            "Bachelor's": [0, 0],
            "Master's": [0, 0],
            "Doctorate": [0, 0]
        }
        
        education_counts = {level: 0 for level in education_impact.keys()}
        
        for person in self.economy.persons:
            if person.education_level in education_impact:
                education_impact[person.education_level][0] += person.monthly_income * 12
                education_impact[person.education_level][1] += person.net_worth
                education_counts[person.education_level] += 1
                
        # Calculate averages
        for level in education_impact:
            if education_counts[level] > 0:
                education_impact[level][0] /= education_counts[level]  # Average income
                education_impact[level][1] /= education_counts[level]  # Average net worth
                
        # Prepare data for plotting
        education_levels = list(education_impact.keys())
        income_data = [education_impact[level][0] for level in education_levels]
        networth_data = [education_impact[level][1] for level in education_levels]
        
        # Scale net worth down for better visualization
        scaled_networth = [value / 10 for value in networth_data]
        
        education_metrics_data = {
            "Annual Income": income_data,
            "Net Worth (÷10)": scaled_networth
        }
        
        education_metrics_chart = self.create_multi_bar_chart(education_levels, education_metrics_data, "Education Impact on Economic Outcomes", "Education Level", "Value (£)")
        layout.addWidget(education_metrics_chart, 1, 1)
        
        return widget
        
    def create_housing_charts(self):
        """Create housing charts"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Housing status distribution
        status_counts = {"Renter": 0, "Owner": 0, "Living with family": 0}
        
        for person in self.economy.persons:
            status_counts[person.housing_status] = status_counts.get(person.housing_status, 0) + 1
            
        # Normalize to percentages
        population = len(self.economy.persons)
        status_percentages = [count / population * 100 for count in status_counts.values()] if population > 0 else [0] * len(status_counts)
        
        status_chart = self.create_bar_chart(list(status_counts.keys()), status_percentages, "Housing Status Distribution", "Status", "Percentage (%)")
        layout.addWidget(status_chart, 0, 0)
        
        # Housing status by age group
        age_groups = ["18-34", "35-54", "55+"]
        
        housing_by_age = {
            age_group: {status: 0 for status in status_counts.keys()}
            for age_group in age_groups
        }
        
        age_group_counts = {group: 0 for group in age_groups}
        
        for person in self.economy.persons:
            if person.age < 35:
                age_group = "18-34"
            elif person.age < 55:
                age_group = "35-54"
            else:
                age_group = "55+"
                
            housing_by_age[age_group][person.housing_status] += 1
            age_group_counts[age_group] += 1
            
        # Calculate percentages
        housing_age_data = {}
        for status in status_counts.keys():
            housing_age_data[status] = []
            for group in age_groups:
                percentage = housing_by_age[group][status] / age_group_counts[group] * 100 if age_group_counts[group] > 0 else 0
                housing_age_data[status].append(percentage)
                
        housing_age_chart = self.create_multi_bar_chart(age_groups, housing_age_data, "Housing Status by Age Group", "Age Group", "Percentage (%)")
        layout.addWidget(housing_age_chart, 0, 1)
        
        # Housing status by income group
        income_groups = ["Low Income", "Middle Income", "High Income"]
        
        # Determine income boundaries
        incomes = [person.monthly_income * 12 for person in self.economy.persons]
        incomes.sort()
        
        # Define low/middle/high as bottom 25%, middle 50%, top 25%
        low_threshold = incomes[int(population * 0.25)] if population > 0 else 0
        high_threshold = incomes[int(population * 0.75)] if population > 0 else 0
        
        housing_by_income = {
            income_group: {status: 0 for status in status_counts.keys()}
            for income_group in income_groups
        }
        
        income_group_counts = {group: 0 for group in income_groups}
        
        for person in self.economy.persons:
            annual_income = person.monthly_income * 12
            if annual_income < low_threshold:
                income_group = "Low Income"
            elif annual_income < high_threshold:
                income_group = "Middle Income"
            else:
                income_group = "High Income"
                
            housing_by_income[income_group][person.housing_status] += 1
            income_group_counts[income_group] += 1
            
        # Calculate percentages
        housing_income_data = {}
        for status in status_counts.keys():
            housing_income_data[status] = []
            for group in income_groups:
                percentage = housing_by_income[group][status] / income_group_counts[group] * 100 if income_group_counts[group] > 0 else 0
                housing_income_data[status].append(percentage)
                
        housing_income_chart = self.create_multi_bar_chart(income_groups, housing_income_data, "Housing Status by Income Group", "Income Group", "Percentage (%)")
        layout.addWidget(housing_income_chart, 1, 0)
        
        # Housing value distribution
        housing_values = [person.housing_value for person in self.economy.persons if person.housing_status == "Owner" and person.housing_value > 0]
        
        if housing_values:
            ranges = [
                "0-100k", "100k-200k", "200k-300k", "300k-400k",
                "400k-500k", "500k-750k", "750k-1M", "1M+"
            ]
            
            value_counts = [0] * len(ranges)
            
            for value in housing_values:
                if value < 100000:
                    value_counts[0] += 1
                elif value < 200000:
                    value_counts[1] += 1
                elif value < 300000:
                    value_counts[2] += 1
                elif value < 400000:
                    value_counts[3] += 1
                elif value < 500000:
                    value_counts[4] += 1
                elif value < 750000:
                    value_counts[5] += 1
                elif value < 1000000:
                    value_counts[6] += 1
                else:
                    value_counts[7] += 1
                    
            # Normalize to percentages
            value_percentages = [count / len(housing_values) * 100 for count in value_counts]
            
            value_chart = self.create_bar_chart(ranges, value_percentages, "Housing Value Distribution", "Value Range", "Percentage (%)")
        else:
            # Create empty chart if no housing values available
            value_chart = self.create_bar_chart(["No Data"], [0], "Housing Value Distribution", "Value Range", "Percentage (%)")
            
        layout.addWidget(value_chart, 1, 1)
        
        return widget
        
    def create_bar_chart(self, categories, values, title, xlabel, ylabel):
        """Create a bar chart"""
        fig = Figure(figsize=(5, 4), dpi=100, facecolor=Config.COLORS["background"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(Config.COLORS["background"])
        
        bars = ax.bar(categories, values, color=Config.COLORS["chart_line"])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        color=Config.COLORS["text_secondary"])
        
        ax.set_title(title, color=Config.COLORS["text_primary"])
        ax.set_xlabel(xlabel, color=Config.COLORS["text_secondary"])
        ax.set_ylabel(ylabel, color=Config.COLORS["text_secondary"])
        
        ax.tick_params(axis='x', colors=Config.COLORS["text_secondary"], rotation=45)
        ax.tick_params(axis='y', colors=Config.COLORS["text_secondary"])
        
        ax.grid(True, linestyle='--', alpha=0.3, color=Config.COLORS["chart_grid"])
        
        fig.tight_layout()
        
        canvas = FigureCanvas(fig)
        return canvas
        
    def create_multi_bar_chart(self, categories, data_dict, title, xlabel, ylabel):
        """Create a multi-bar chart"""
        fig = Figure(figsize=(5, 4), dpi=100, facecolor=Config.COLORS["background"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(Config.COLORS["background"])
        
        num_categories = len(categories)
        num_data_series = len(data_dict)
        bar_width = 0.8 / num_data_series
        
        # Plot each data series
        for i, (label, values) in enumerate(data_dict.items()):
            x = np.arange(num_categories)
            offset = (i - num_data_series / 2 + 0.5) * bar_width
            
            bars = ax.bar(x + offset, values, 
                          width=bar_width, 
                          label=label, 
                          color=Config.CHART_COLORS[i % len(Config.CHART_COLORS)])
        
        ax.set_title(title, color=Config.COLORS["text_primary"])
        ax.set_xlabel(xlabel, color=Config.COLORS["text_secondary"])
        ax.set_ylabel(ylabel, color=Config.COLORS["text_secondary"])
        
        ax.set_xticks(np.arange(num_categories))
        ax.set_xticklabels(categories)
        
        ax.tick_params(axis='x', colors=Config.COLORS["text_secondary"])
        ax.tick_params(axis='y', colors=Config.COLORS["text_secondary"])
        
        ax.grid(True, linestyle='--', alpha=0.3, color=Config.COLORS["chart_grid"])
        
        ax.legend(facecolor=Config.COLORS["surface_light"], edgecolor=Config.COLORS["border"], labelcolor=Config.COLORS["text_primary"])
        
        fig.tight_layout()
        
        canvas = FigureCanvas(fig)
        return canvas

class IndustryStatsDialog(QDialog):
    """Dialog showing detailed industry statistics"""
    def __init__(self, economy, parent=None):
        super().__init__(parent)
        self.economy = economy
        
        self.setWindowTitle("Industry Statistics")
        self.setMinimumSize(900, 700)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Industry Statistics")
        title_label.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {Config.COLORS['primary']};")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Main tabs
        tabs = QTabWidget()
        
        # Sector performance tab
        sector_tab = QWidget()
        sector_layout = QVBoxLayout(sector_tab)
        
        # Create sector performance charts
        sector_layout.addWidget(self.create_sector_performance_charts())
        
        tabs.addTab(sector_tab, "Sector Performance")
        
        # Company metrics tab
        company_tab = QWidget()
        company_layout = QVBoxLayout(company_tab)
        
        # Create company metrics charts
        company_layout.addWidget(self.create_company_metrics_charts())
        
        tabs.addTab(company_tab, "Company Metrics")
        
        # Market structure tab
        market_tab = QWidget()
        market_layout = QVBoxLayout(market_tab)
        
        # Create market structure charts
        market_layout.addWidget(self.create_market_structure_charts())
        
        tabs.addTab(market_tab, "Market Structure")
        
        # Innovation tab
        innovation_tab = QWidget()
        innovation_layout = QVBoxLayout(innovation_tab)
        
        # Create innovation charts
        innovation_layout.addWidget(self.create_innovation_charts())
        
        tabs.addTab(innovation_tab, "Innovation & R&D")
        
        layout.addWidget(tabs)
        
        # Summary statistics
        summary_group = QGroupBox("Industry Summary")
        summary_layout = QGridLayout(summary_group)
        
        # Calculate summary stats
        total_companies = len(self.economy.companies)
        total_revenue = sum(company.revenue for company in self.economy.companies) * 12  # Annual
        total_profit = sum(max(0, company.profit) for company in self.economy.companies) * 12  # Annual
        total_employees = sum(len(company.employees) for company in self.economy.companies)
        avg_company_size = total_employees / total_companies if total_companies > 0 else 0
        profit_margin = total_profit / total_revenue if total_revenue > 0 else 0
        
        # Public companies stats
        public_companies = sum(1 for company in self.economy.companies if company.is_public)
        public_market_cap = sum(company.value for company in self.economy.companies if company.is_public)
        
        # Add summary stats to grid
        summary_layout.addWidget(QLabel("Total Companies:"), 0, 0)
        summary_layout.addWidget(QLabel(f"{total_companies:,}"), 0, 1)
        
        summary_layout.addWidget(QLabel("Total Revenue:"), 0, 2)
        summary_layout.addWidget(QLabel(format_currency(total_revenue)), 0, 3)
        
        summary_layout.addWidget(QLabel("Total Profit:"), 1, 0)
        summary_layout.addWidget(QLabel(format_currency(total_profit)), 1, 1)
        
        summary_layout.addWidget(QLabel("Profit Margin:"), 1, 2)
        summary_layout.addWidget(QLabel(f"{profit_margin*100:.1f}%"), 1, 3)
        
        summary_layout.addWidget(QLabel("Total Employees:"), 2, 0)
        summary_layout.addWidget(QLabel(f"{total_employees:,}"), 2, 1)
        
        summary_layout.addWidget(QLabel("Average Company Size:"), 2, 2)
        summary_layout.addWidget(QLabel(f"{avg_company_size:.1f} employees"), 2, 3)
        
        summary_layout.addWidget(QLabel("Public Companies:"), 3, 0)
        summary_layout.addWidget(QLabel(f"{public_companies:,} ({public_companies/total_companies*100:.1f}%)"), 3, 1)
        
        summary_layout.addWidget(QLabel("Public Market Cap:"), 3, 2)
        summary_layout.addWidget(QLabel(format_currency(public_market_cap)), 3, 3)
        
        layout.addWidget(summary_group)
        
        # Close button
        close_button = ModernPushButton("Close", False)
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
    
    def create_sector_performance_charts(self):
        """Create sector performance charts"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Sector distribution
        sector_counts = {}
        for company in self.economy.companies:
            sector_counts[company.sector] = sector_counts.get(company.sector, 0) + 1
            
        # Sort sectors by count
        sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
        sectors = [s[0] for s in sorted_sectors]
        counts = [s[1] for s in sorted_sectors]
        
        # Calculate percentages
        total_companies = len(self.economy.companies)
        percentages = [count / total_companies * 100 for count in counts] if total_companies > 0 else [0] * len(counts)
        
        sector_chart = self.create_bar_chart(sectors, percentages, "Companies by Sector", "Sector", "Percentage (%)")
        layout.addWidget(sector_chart, 0, 0)
        
        # Sector revenue
        sector_revenue = {}
        for company in self.economy.companies:
            sector_revenue[company.sector] = sector_revenue.get(company.sector, 0) + (company.revenue * 12)  # Annual revenue
            
        # Sort sectors by revenue
        sorted_revenue = sorted(sector_revenue.items(), key=lambda x: x[1], reverse=True)
        revenue_sectors = [s[0] for s in sorted_revenue]
        revenues = [s[1] / 1e9 for s in sorted_revenue]  # In billions for better visualization
        
        revenue_chart = self.create_bar_chart(revenue_sectors, revenues, "Sector Revenue (Annual)", "Sector", "Revenue (£B)")
        layout.addWidget(revenue_chart, 0, 1)
        
        # Sector profitability
        sector_profit = {}
        sector_revenue_for_margin = {}
        for company in self.economy.companies:
            sector_profit[company.sector] = sector_profit.get(company.sector, 0) + (company.profit * 12)  # Annual profit
            sector_revenue_for_margin[company.sector] = sector_revenue_for_margin.get(company.sector, 0) + (company.revenue * 12)
            
        # Calculate profit margins by sector
        profit_margins = {}
        for sector in sector_profit:
            if sector_revenue_for_margin[sector] > 0:
                profit_margins[sector] = sector_profit[sector] / sector_revenue_for_margin[sector] * 100  # As percentage
            else:
                profit_margins[sector] = 0
                
        # Sort sectors by profit margin
        sorted_margins = sorted(profit_margins.items(), key=lambda x: x[1], reverse=True)
        margin_sectors = [s[0] for s in sorted_margins]
        margins = [s[1] for s in sorted_margins]
        
        margin_chart = self.create_bar_chart(margin_sectors, margins, "Sector Profit Margins", "Sector", "Profit Margin (%)")
        layout.addWidget(margin_chart, 1, 0)
        
        # Sector employment
        sector_employment = {}
        for company in self.economy.companies:
            sector_employment[company.sector] = sector_employment.get(company.sector, 0) + len(company.employees)
            
        # Sort sectors by employment
        sorted_employment = sorted(sector_employment.items(), key=lambda x: x[1], reverse=True)
        employment_sectors = [s[0] for s in sorted_employment]
        employment = [s[1] for s in sorted_employment]
        
        employment_chart = self.create_bar_chart(employment_sectors, employment, "Sector Employment", "Sector", "Employees")
        layout.addWidget(employment_chart, 1, 1)
        
        return widget
        
    def create_company_metrics_charts(self):
        """Create company metrics charts"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Company size distribution
        size_ranges = ["1-10", "11-50", "51-100", "101-500", "500+"]
        size_counts = [0] * len(size_ranges)
        
        for company in self.economy.companies:
            employees = len(company.employees)
            if employees <= 10:
                size_counts[0] += 1
            elif employees <= 50:
                size_counts[1] += 1
            elif employees <= 100:
                size_counts[2] += 1
            elif employees <= 500:
                size_counts[3] += 1
            else:
                size_counts[4] += 1
                
        # Calculate percentages
        total_companies = len(self.economy.companies)
        size_percentages = [count / total_companies * 100 for count in size_counts] if total_companies > 0 else [0] * len(size_ranges)
        
        size_chart = self.create_bar_chart(size_ranges, size_percentages, "Company Size Distribution", "Number of Employees", "Percentage (%)")
        layout.addWidget(size_chart, 0, 0)
        
        # Revenue distribution
        revenue_ranges = ["0-100K", "100K-1M", "1M-10M", "10M-100M", "100M+"]
        revenue_counts = [0] * len(revenue_ranges)
        
        for company in self.economy.companies:
            annual_revenue = company.revenue * 12
            if annual_revenue < 100000:
                revenue_counts[0] += 1
            elif annual_revenue < 1000000:
                revenue_counts[1] += 1
            elif annual_revenue < 10000000:
                revenue_counts[2] += 1
            elif annual_revenue < 100000000:
                revenue_counts[3] += 1
            else:
                revenue_counts[4] += 1
                
        # Calculate percentages
        revenue_percentages = [count / total_companies * 100 for count in revenue_counts] if total_companies > 0 else [0] * len(revenue_ranges)
        
        revenue_chart = self.create_bar_chart(revenue_ranges, revenue_percentages, "Company Revenue Distribution", "Annual Revenue", "Percentage (%)")
        layout.addWidget(revenue_chart, 0, 1)
        
        # Profitability distribution
        profitability_ranges = ["Loss", "0-5%", "5-10%", "10-20%", "20%+"]
        profitability_counts = [0] * len(profitability_ranges)
        
        for company in self.economy.companies:
            if company.revenue > 0:
                profit_margin = company.profit / company.revenue
            else:
                profit_margin = 0
                
            if profit_margin < 0:
                profitability_counts[0] += 1
            elif profit_margin < 0.05:
                profitability_counts[1] += 1
            elif profit_margin < 0.1:
                profitability_counts[2] += 1
            elif profit_margin < 0.2:
                profitability_counts[3] += 1
            else:
                profitability_counts[4] += 1
                
        # Calculate percentages
        profitability_percentages = [count / total_companies * 100 for count in profitability_counts] if total_companies > 0 else [0] * len(profitability_ranges)
        
        profitability_chart = self.create_bar_chart(profitability_ranges, profitability_percentages, "Company Profitability Distribution", "Profit Margin", "Percentage (%)")
        layout.addWidget(profitability_chart, 1, 0)
        
        # Company age distribution
        age_ranges = ["0-1 year", "1-3 years", "3-5 years", "5-10 years", "10+ years"]
        age_counts = [0] * len(age_ranges)
        
        for company in self.economy.companies:
            age_years = company.age / 12  # Convert months to years
            if age_years < 1:
                age_counts[0] += 1
            elif age_years < 3:
                age_counts[1] += 1
            elif age_years < 5:
                age_counts[2] += 1
            elif age_years < 10:
                age_counts[3] += 1
            else:
                age_counts[4] += 1
                
        # Calculate percentages
        age_percentages = [count / total_companies * 100 for count in age_counts] if total_companies > 0 else [0] * len(age_ranges)
        
        age_chart = self.create_bar_chart(age_ranges, age_percentages, "Company Age Distribution", "Company Age", "Percentage (%)")
        layout.addWidget(age_chart, 1, 1)
        
        return widget
        
    def create_market_structure_charts(self):
        """Create market structure charts"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Market concentration by sector
        sectors = set(company.sector for company in self.economy.companies)
        concentration_data = []
        
        for sector in sectors:
            # Get companies in this sector
            sector_companies = [company for company in self.economy.companies if company.sector == sector]
            
            if not sector_companies:
                continue
                
            # Calculate total sector revenue
            sector_revenue = sum(company.revenue for company in sector_companies)
            
            # Sort companies by revenue
            sorted_companies = sorted(sector_companies, key=lambda x: x.revenue, reverse=True)
            
            # Calculate concentration ratio (CR4) - share of top 4 companies
            top4_revenue = sum(company.revenue for company in sorted_companies[:4])
            cr4 = top4_revenue / sector_revenue if sector_revenue > 0 else 0
            
            concentration_data.append((sector, cr4 * 100))  # As percentage
            
        # Sort sectors by concentration
        concentration_data.sort(key=lambda x: x[1], reverse=True)
        
        # Extract data for chart
        concentration_sectors = [data[0] for data in concentration_data]
        concentration_values = [data[1] for data in concentration_data]
        
        concentration_chart = self.create_bar_chart(concentration_sectors, concentration_values, "Market Concentration by Sector (CR4)", "Sector", "Concentration Ratio (%)")
        layout.addWidget(concentration_chart, 0, 0)
        
        # Public vs private companies by sector
        public_private_data = {}
        
        for sector in sectors:
            # Get companies in this sector
            sector_companies = [company for company in self.economy.companies if company.sector == sector]
            
            if len(sector_companies) < 5:  # Skip sectors with few companies
                continue
                
            # Count public and private companies
            public_count = sum(1 for company in sector_companies if company.is_public)
            private_count = len(sector_companies) - public_count
            
            # Calculate percentages
            public_pct = public_count / len(sector_companies) * 100 if sector_companies else 0
            private_pct = 100 - public_pct
            
            public_private_data[sector] = [public_pct, private_pct]
            
        # Sort sectors by public percentage
        sorted_sectors = sorted(public_private_data.items(), key=lambda x: x[1][0], reverse=True)
        pp_sectors = [data[0] for data in sorted_sectors]
        
        # Prepare data for stacked bar chart
        public_data = {
            "Public": [data[1][0] for data in sorted_sectors],
            "Private": [data[1][1] for data in sorted_sectors]
        }
        
        public_private_chart = self.create_stacked_bar_chart(pp_sectors, public_data, "Public vs Private Companies by Sector", "Sector", "Percentage (%)")
        layout.addWidget(public_private_chart, 0, 1)
        
        # Business type distribution
        business_type_counts = {}
        for company in self.economy.companies:
            business_type_counts[company.business_type] = business_type_counts.get(company.business_type, 0) + 1
            
        # Sort business types by count
        sorted_types = sorted(business_type_counts.items(), key=lambda x: x[1], reverse=True)
        business_types = [t[0] for t in sorted_types][:10]  # Top 10 business types
        type_counts = [t[1] for t in sorted_types][:10]
        
        # Calculate percentages
        try:
            type_percentages = [count / sorted_companies * 100 for count in type_counts] if sorted_companies > 0 else [0] * len(business_types)
        except:
            type_percentages = 0
        business_type_chart = self.create_bar_chart(business_types, type_percentages, "Top 10 Business Types", "Business Type", "Percentage (%)")
        layout.addWidget(business_type_chart, 1, 0)
        
        # Company growth distribution
        growth_ranges = ["Negative", "0-5%", "5-10%", "10-20%", "20%+"]
        growth_counts = [0] * len(growth_ranges)
        
        for company in self.economy.companies:
            if hasattr(company, 'growth_rate'):
                growth_rate = company.growth_rate * 100  # As percentage
                if growth_rate < 0:
                    growth_counts[0] += 1
                elif growth_rate < 5:
                    growth_counts[1] += 1
                elif growth_rate < 10:
                    growth_counts[2] += 1
                elif growth_rate < 20:
                    growth_counts[3] += 1
                else:
                    growth_counts[4] += 1
                    
        # Calculate percentages
        growth_percentages = [count / total_companies * 100 for count in growth_counts] if total_companies > 0 else [0] * len(growth_ranges)
        
        growth_chart = self.create_bar_chart(growth_ranges, growth_percentages, "Company Growth Rate Distribution", "Growth Rate", "Percentage (%)")
        layout.addWidget(growth_chart, 1, 1)
        
        return widget
        
    def create_innovation_charts(self):
        """Create innovation and R&D charts"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # R&D spending by sector
        sector_r_and_d = {}
        sector_revenue = {}
        
        for company in self.economy.companies:
            if hasattr(company, 'r_and_d_investment'):
                sector_r_and_d[company.sector] = sector_r_and_d.get(company.sector, 0) + (company.r_and_d_investment * 12)  # Annual
                sector_revenue[company.sector] = sector_revenue.get(company.sector, 0) + (company.revenue * 12)  # Annual
                
        # Calculate R&D intensity (R&D as % of revenue)
        r_and_d_intensity = {}
        for sector in sector_r_and_d:
            if sector_revenue[sector] > 0:
                r_and_d_intensity[sector] = sector_r_and_d[sector] / sector_revenue[sector] * 100  # As percentage
            else:
                r_and_d_intensity[sector] = 0
                
        # Sort sectors by R&D intensity
        sorted_intensity = sorted(r_and_d_intensity.items(), key=lambda x: x[1], reverse=True)
        intensity_sectors = [s[0] for s in sorted_intensity]
        intensity_values = [s[1] for s in sorted_intensity]
        
        intensity_chart = self.create_bar_chart(intensity_sectors, intensity_values, "R&D Intensity by Sector (% of Revenue)", "Sector", "R&D Intensity (%)")
        layout.addWidget(intensity_chart, 0, 0)
        
        # Innovation level distribution
        innovation_ranges = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
        innovation_counts = [0] * len(innovation_ranges)
        
        for company in self.economy.companies:
            if hasattr(company, 'innovation_level'):
                innovation = company.innovation_level * 100  # As percentage
                idx = min(int(innovation / 20), 4)
                innovation_counts[idx] += 1
                
        # Calculate percentages
        total_companies = len(self.economy.companies)
        innovation_percentages = [count / total_companies * 100 for count in innovation_counts] if total_companies > 0 else [0] * len(innovation_ranges)
        
        innovation_chart = self.create_bar_chart(innovation_ranges, innovation_percentages, "Company Innovation Level Distribution", "Innovation Level", "Percentage (%)")
        layout.addWidget(innovation_chart, 0, 1)
        
        # Technology level by sector
        sector_technology = {}
        sector_company_counts = {}
        
        for company in self.economy.companies:
            if hasattr(company, 'technology_level'):
                sector_technology[company.sector] = sector_technology.get(company.sector, 0) + company.technology_level
                sector_company_counts[company.sector] = sector_company_counts.get(company.sector, 0) + 1
                
        # Calculate average technology level by sector
        avg_technology = {}
        for sector in sector_technology:
            if sector_company_counts[sector] > 0:
                avg_technology[sector] = sector_technology[sector] / sector_company_counts[sector] * 100  # As percentage
            else:
                avg_technology[sector] = 0
                
        # Sort sectors by average technology level
        sorted_tech = sorted(avg_technology.items(), key=lambda x: x[1], reverse=True)
        tech_sectors = [s[0] for s in sorted_tech]
        tech_values = [s[1] for s in sorted_tech]
        
        technology_chart = self.create_bar_chart(tech_sectors, tech_values, "Average Technology Level by Sector", "Sector", "Technology Level (%)")
        layout.addWidget(technology_chart, 1, 0)
        
        # R&D spending vs profitability
        if len(self.economy.companies) > 0:
            # Create scatter plot
            r_and_d_vs_profit_chart = self.create_scatter_chart(
                "R&D Investment vs Profitability",
                "R&D Intensity (%)",
                "Profit Margin (%)"
            )
            layout.addWidget(r_and_d_vs_profit_chart, 1, 1)
        
        return widget
        
    def create_bar_chart(self, categories, values, title, xlabel, ylabel):
        """Create a bar chart"""
        fig = Figure(figsize=(5, 4), dpi=100, facecolor=Config.COLORS["background"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(Config.COLORS["background"])
        
        bars = ax.bar(categories, values, color=Config.COLORS["chart_line"])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        color=Config.COLORS["text_secondary"])
        
        ax.set_title(title, color=Config.COLORS["text_primary"])
        ax.set_xlabel(xlabel, color=Config.COLORS["text_secondary"])
        ax.set_ylabel(ylabel, color=Config.COLORS["text_secondary"])
        
        ax.tick_params(axis='x', colors=Config.COLORS["text_secondary"], rotation=45)
        ax.tick_params(axis='y', colors=Config.COLORS["text_secondary"])
        
        ax.grid(True, linestyle='--', alpha=0.3, color=Config.COLORS["chart_grid"])
        
        fig.tight_layout()
        
        canvas = FigureCanvas(fig)
        return canvas
        
    def create_stacked_bar_chart(self, categories, data_dict, title, xlabel, ylabel):
        """Create a stacked bar chart"""
        fig = Figure(figsize=(5, 4), dpi=100, facecolor=Config.COLORS["background"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(Config.COLORS["background"])
        
        bottom = np.zeros(len(categories))
        for i, (label, values) in enumerate(data_dict.items()):
            ax.bar(categories, values, bottom=bottom, label=label, color=Config.CHART_COLORS[i % len(Config.CHART_COLORS)])
            bottom += np.array(values)
            
        ax.set_title(title, color=Config.COLORS["text_primary"])
        ax.set_xlabel(xlabel, color=Config.COLORS["text_secondary"])
        ax.set_ylabel(ylabel, color=Config.COLORS["text_secondary"])
        
        ax.tick_params(axis='x', colors=Config.COLORS["text_secondary"], rotation=45)
        ax.tick_params(axis='y', colors=Config.COLORS["text_secondary"])
        
        ax.grid(True, linestyle='--', alpha=0.3, color=Config.COLORS["chart_grid"])
        
        ax.legend(facecolor=Config.COLORS["surface_light"], edgecolor=Config.COLORS["border"], labelcolor=Config.COLORS["text_primary"])
        
        fig.tight_layout()
        
        canvas = FigureCanvas(fig)
        return canvas
        
    def create_scatter_chart(self, title, xlabel, ylabel):
        """Create a scatter chart for R&D vs profitability"""
        fig = Figure(figsize=(5, 4), dpi=100, facecolor=Config.COLORS["background"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(Config.COLORS["background"])
        
        # Collect data points
        x_values = []  # R&D intensity
        y_values = []  # Profit margin
        sizes = []     # Company size (employees)
        colors = []    # Sector color
        
        # Create mapping of sectors to colors
        sectors = list(set(company.sector for company in self.economy.companies))
        sector_colors = {sector: Config.CHART_COLORS[i % len(Config.CHART_COLORS)] for i, sector in enumerate(sectors)}
        
        for company in self.economy.companies:
            if (hasattr(company, 'r_and_d_investment') and company.revenue > 0):
                r_and_d_intensity = company.r_and_d_investment / company.revenue * 100  # As percentage
                profit_margin = company.profit / company.revenue * 100 if company.profit > 0 else 0  # As percentage
                
                x_values.append(r_and_d_intensity)
                y_values.append(profit_margin)
                
                # Size based on number of employees (sqrt for better visualization)
                size = max(20, np.sqrt(len(company.employees)) * 5)
                sizes.append(size)
                
                # Color based on sector
                colors.append(sector_colors.get(company.sector, Config.COLORS["chart_line"]))
        
        # Create scatter plot
        if x_values:
            ax.scatter(x_values, y_values, s=sizes, c=colors, alpha=0.6)
            
            # Add trendline
            if len(x_values) > 1:
                z = np.polyfit(x_values, y_values, 1)
                p = np.poly1d(z)
                ax.plot(x_values, p(x_values), "r--", color="white", alpha=0.8)
        
        ax.set_title(title, color=Config.COLORS["text_primary"])
        ax.set_xlabel(xlabel, color=Config.COLORS["text_secondary"])
        ax.set_ylabel(ylabel, color=Config.COLORS["text_secondary"])
        
        ax.tick_params(axis='x', colors=Config.COLORS["text_secondary"])
        ax.tick_params(axis='y', colors=Config.COLORS["text_secondary"])
        
        ax.grid(True, linestyle='--', alpha=0.3, color=Config.COLORS["chart_grid"])
        
        fig.tight_layout()
        
        canvas = FigureCanvas(fig)
        return canvas

# -----------------------------
# Main entry point
# -----------------------------
def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)
    
    # Create splash screen to show progress
    splash = SplashScreen()
    splash.show()
    QApplication.processEvents()
    
    # Create the economy
    splash.update_progress(10, "Creating economy...", "Initializing economic framework")
    economy = Economy(num_persons=Config.DEFAULT_NUM_PERSONS, num_companies=Config.DEFAULT_NUM_COMPANIES)
    
    # Initialize persons
    splash.update_progress(20, "Creating population...", "Generating individual citizens")
    economy.initialize_people()
    
    # Initialize companies
    splash.update_progress(60, "Creating companies...", "Establishing business entities")
    economy.initialize_companies()
    

        
    splash.update_progress(95, "Finalizing...", "Preparing user interface")
    
    # Create and show the main window
    main_window = MainWindow(economy)
    
    # Finish splash and show main window
    splash.update_progress(100, "Ready!", "Application loaded successfully")
    splash.finish(main_window)
    main_window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()




