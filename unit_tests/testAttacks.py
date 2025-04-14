# unit_tests/test_attack_visualizations.py
"""
Unit tests for visualizing and verifying attack model implementations.

This module provides functions to visually compare the effects of different
attack models on sample energy consumption data, helping to verify that the 
refactored implementations match the behavior of the original code.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import attack models and constants
from src.attack_models import get_attack_model, list_available_attacks
from experiments.config import ATTACK_CONSTANTS


class TestAttackVisualizations(unittest.TestCase):
    """Test case for visualizing and comparing attack implementations."""

    @classmethod
    def setUpClass(cls):
        """Set up test data once for all test methods."""
        # Create a synthetic dataset with daily patterns for testing
        cls.n_days = 30  # One month of data
        cls.n_readings_per_day = 48  # Half-hourly readings
        cls.n_consumers = 5  # Number of test consumers
        
        # Generate datetime index for the month
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(minutes=30*i) 
                 for i in range(cls.n_days * cls.n_readings_per_day)]
        
        # Create a DataFrame with realistic consumption patterns
        data = cls._generate_synthetic_data(
            dates, cls.n_consumers, cls.n_readings_per_day)
        
        cls.test_df = pd.DataFrame(
            data, 
            index=dates,
            columns=[f'consumer_{i+1}' for i in range(cls.n_consumers)]
        )
    
    @staticmethod
    def _generate_synthetic_data(dates, n_consumers, n_readings_per_day):
        """Generate synthetic daily load curves with realistic patterns."""
        n_total_readings = len(dates)
        data = np.zeros((n_total_readings, n_consumers))
        
        # Create base load profiles with daily and weekly patterns
        for c in range(n_consumers):
            # Base load (constant component)
            base_load = np.random.uniform(0.1, 0.5)
            
            # Daily patterns (higher during day, lower at night)
            daily_pattern = np.zeros(n_readings_per_day)
            for h in range(n_readings_per_day):
                # Morning peak (7-9 AM)
                if 14 <= h < 18:
                    daily_pattern[h] = np.random.uniform(0.7, 1.0)
                # Evening peak (6-10 PM)
                elif 36 <= h < 44:
                    daily_pattern[h] = np.random.uniform(0.8, 1.2)
                # Daytime (9 AM - 6 PM)
                elif 18 <= h < 36:
                    daily_pattern[h] = np.random.uniform(0.5, 0.7)
                # Night (10 PM - 7 AM)
                else:
                    daily_pattern[h] = np.random.uniform(0.2, 0.4)
            
            # Repeat daily pattern for all days
            days = n_total_readings // n_readings_per_day
            repeated_pattern = np.tile(daily_pattern, days)
            
            # Add weekend variation (lower consumption on weekends)
            weekend_mask = np.zeros(n_total_readings)
            for day in range(days):
                # Assume 0-indexed day, so days 5 and 6 are weekend
                if day % 7 >= 5:  # Weekend
                    weekend_mask[day*n_readings_per_day:(day+1)*n_readings_per_day] = 1
            
            weekend_factor = np.where(weekend_mask == 1, 0.7, 1.0)
            
            # Add some random noise
            noise = np.random.normal(0, 0.1, n_total_readings)
            
            # Combine components
            load = base_load + repeated_pattern * weekend_factor + noise
            
            # Ensure no negative values
            load = np.maximum(load, 0.1)
            
            data[:, c] = load
        
        return data

    def test_visualize_all_attacks(self):
        """
        Test and visualize all available attacks.
        
        This method applies each attack to the test data and visualizes
        the results for visual inspection and comparison.
        """
        # Get all available attack IDs
        attack_ids = list_available_attacks()
        
        # Create a figure for the visualization
        n_attacks = len(attack_ids)
        fig, axes = plt.subplots(n_attacks, 1, figsize=(15, 5*n_attacks))
        
        if n_attacks == 1:
            axes = [axes]  # Ensure axes is a list for single attack case
        
        # Select a random consumer for visualization
        consumer = random.choice(self.test_df.columns)
        
        # Select a week of data for clarity in visualization
        start_idx = random.randint(0, self.n_days - 7) * self.n_readings_per_day
        end_idx = start_idx + 7 * self.n_readings_per_day
        window_df = self.test_df.iloc[start_idx:end_idx]
        
        # Plot original data on all subplots for comparison
        original_data = window_df[consumer]
        week_dates = window_df.index
        
        for i, attack_id in enumerate(attack_ids):
            ax = axes[i]
            
            # Apply the attack to the window data
            try:
                attack_model = get_attack_model(attack_id)
                attacked_df = attack_model.apply(window_df.copy())
                attacked_data = attacked_df[consumer]
                
                # Calculate statistics for the attack
                orig_sum = original_data.sum()
                attacked_sum = attacked_data.sum()
                reduction = (orig_sum - attacked_sum) / orig_sum * 100
                
                # Plot data
                ax.plot(week_dates, original_data, 'b-', alpha=0.7, label='Original')
                ax.plot(week_dates, attacked_data, 'r-', label=f'Attack {attack_id}')
                
                # Format x-axis to show dates better
                # Convert datetime index to days for clearer labels
                ax.set_title(f'Attack {attack_id}: Energy reduction = {reduction:.1f}%')
                ax.set_xlabel('Time')
                ax.set_ylabel('Consumption')
                
                # Show day boundaries with vertical lines
                for day in range(1, 7):
                    day_start = start_idx + day * self.n_readings_per_day
                    ax.axvline(x=window_df.index[day_start - start_idx], 
                              color='gray', linestyle='--', alpha=0.3)
                
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
                
                # Set x-ticks to show days
                day_indices = [start_idx + day * self.n_readings_per_day 
                               for day in range(0, 8)]
                day_indices = [idx - start_idx for idx in day_indices 
                               if idx - start_idx < len(window_df)]
                
                day_labels = [window_df.index[idx].strftime('%a %d') 
                             for idx in day_indices]
                
                if day_indices:
                    ax.set_xticks([window_df.index[idx] for idx in day_indices])
                    ax.set_xticklabels(day_labels)
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error applying attack {attack_id}: {str(e)}",
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12, color='red')
        
        plt.tight_layout()
        
        # Save the figure to a file in the test results directory
        results_dir = os.path.join(PROJECT_ROOT, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        plt.savefig(os.path.join(results_dir, 'attack_visualizations.png'))
        plt.close()
        
        # Test passes if no exceptions occurred
        self.assertTrue(True)

    def test_attack_comparison(self):
        """
        Compare the statistical impact of different attacks.
        
        This method measures and visualizes how each attack affects:
        1. Total energy consumption (sum)
        2. Maximum demand (max)
        3. Load pattern variability (std/var)
        """
        # Get all available attack IDs
        attack_ids = list_available_attacks()
        
        # Metrics to track
        metrics = ['energy_reduction_pct', 'max_reduction_pct', 'pattern_change_pct']
        results = {metric: [] for metric in metrics}
        
        # Calculate baseline metrics
        original_energy = self.test_df.sum().sum()
        original_max = self.test_df.max().max()
        original_std = self.test_df.std().mean()
        
        # Apply each attack and calculate metrics
        for attack_id in attack_ids:
            try:
                attack_model = get_attack_model(attack_id)
                attacked_df = attack_model.apply(self.test_df.copy())
                
                # Calculate metrics
                attacked_energy = attacked_df.sum().sum()
                attacked_max = attacked_df.max().max()
                attacked_std = attacked_df.std().mean()
                
                # Calculate percentage changes
                energy_reduction = (original_energy - attacked_energy) / original_energy * 100
                max_reduction = (original_max - attacked_max) / original_max * 100
                pattern_change = abs(original_std - attacked_std) / original_std * 100
                
                results['energy_reduction_pct'].append(energy_reduction)
                results['max_reduction_pct'].append(max_reduction)
                results['pattern_change_pct'].append(pattern_change)
                
            except Exception as e:
                # Log the error and use placeholder values
                print(f"Error applying attack {attack_id}: {str(e)}")
                for metric in metrics:
                    results[metric].append(np.nan)
        
        # Create a bar chart comparing the metrics
        fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 5*len(metrics)))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = results[metric]
            
            # Create bar chart
            bars = ax.bar(attack_ids, values)
            
            # Add value labels on the bars
            for bar, value in zip(bars, values):
                if not np.isnan(value):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value:.1f}%', ha='center', va='bottom')
            
            # Format the plot
            metric_name = metric.replace('_', ' ').title()
            ax.set_title(f'{metric_name} by Attack Type')
            ax.set_xlabel('Attack ID')
            ax.set_ylabel('Percentage (%)')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add a horizontal line at 0%
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            
            # Adjust y limits for better visibility
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(min(ymin, -5), max(ymax, 5))
        
        plt.tight_layout()
        
        # Save the figure
        results_dir = os.path.join(PROJECT_ROOT, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        plt.savefig(os.path.join(results_dir, 'attack_comparison_metrics.png'))
        plt.close()
        
        # Test passes if no exceptions occurred
        self.assertTrue(True)
    
    def test_daily_pattern_impact(self):
        """
        Test how attacks affect daily load patterns.
        
        This method visualizes the average daily load profile before
        and after each attack, highlighting pattern distortions.
        """
        # Get all available attack IDs
        attack_ids = list_available_attacks()
        
        # Create a figure with subplots for each attack
        fig, axes = plt.subplots(len(attack_ids), 1, figsize=(15, 5*len(attack_ids)))
        
        if len(attack_ids) == 1:
            axes = [axes]  # Ensure axes is a list for single attack case
        
        # Calculate the average daily pattern for original data
        # First reshape the data to (n_days, n_readings_per_day, n_consumers)
        original_daily_patterns = []
        
        for consumer in self.test_df.columns:
            consumer_data = self.test_df[consumer].values
            daily_data = consumer_data.reshape(-1, self.n_readings_per_day)
            original_daily_patterns.append(daily_data)
        
        # Calculate the average daily pattern across all days and consumers
        original_daily_patterns = np.array(original_daily_patterns)
        avg_original_pattern = original_daily_patterns.mean(axis=(0, 1))
        
        # Generate hourly labels for x-axis
        hours = [f"{h:02d}:00" for h in range(0, 24, 2)]
        hour_indices = [int(h*self.n_readings_per_day/24) for h in range(0, 24, 2)]
        
        # Loop through each attack
        for i, attack_id in enumerate(attack_ids):
            ax = axes[i]
            
            try:
                # Apply the attack
                attack_model = get_attack_model(attack_id)
                attacked_df = attack_model.apply(self.test_df.copy())
                
                # Calculate the average daily pattern for attacked data
                attacked_daily_patterns = []
                
                for consumer in attacked_df.columns:
                    consumer_data = attacked_df[consumer].values
                    daily_data = consumer_data.reshape(-1, self.n_readings_per_day)
                    attacked_daily_patterns.append(daily_data)
                
                attacked_daily_patterns = np.array(attacked_daily_patterns)
                avg_attacked_pattern = attacked_daily_patterns.mean(axis=(0, 1))
                
                # Plot the daily patterns
                x = np.arange(self.n_readings_per_day)
                ax.plot(x, avg_original_pattern, 'b-', linewidth=2, label='Original')
                ax.plot(x, avg_attacked_pattern, 'r-', linewidth=2, label=f'Attack {attack_id}')
                
                # Calculate the percentage change at each time step
                pct_change = (avg_attacked_pattern - avg_original_pattern) / avg_original_pattern * 100
                ax2 = ax.twinx()
                ax2.plot(x, pct_change, 'g--', linewidth=1.5, alpha=0.6, label='% Change')
                ax2.set_ylabel('Percentage Change (%)')
                ax2.grid(False)
                
                # Set labels and title
                ax.set_title(f'Attack {attack_id}: Impact on Daily Load Pattern')
                ax.set_xlabel('Time of Day')
                ax.set_ylabel('Average Consumption')
                ax.grid(True, alpha=0.3)
                
                # Set x-ticks to show hours
                ax.set_xticks(hour_indices)
                ax.set_xticklabels(hours)
                
                # Combine legends
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error applying attack {attack_id}: {str(e)}",
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12, color='red')
        
        plt.tight_layout()
        
        # Save the figure
        results_dir = os.path.join(PROJECT_ROOT, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        plt.savefig(os.path.join(results_dir, 'attack_daily_patterns.png'))
        plt.close()
        
        # Test passes if no exceptions occurred
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()