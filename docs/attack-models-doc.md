# Attack Models Documentation

This document provides detailed information about all the attack models implemented in this project. These synthetic attack models are designed to simulate different types of energy theft behaviors that might occur in real-world smart grid systems.

## Overview

Energy theft in smart grids can take various forms, from simple manipulation of meter readings to sophisticated modifications of consumption patterns. Since labeled datasets of real-world energy theft are scarce, researchers have developed synthetic attack models to simulate these behaviors.

Our project implements 14 different attack models (numbered 0-13), each representing a different type of energy theft behavior. These models can be applied to honest consumption data to generate synthetic theft examples for training and evaluating detection algorithms.

## Attack Models

### Attack 0: Zero Consumption

**Description**: Reports zero consumption for all readings.

**Mathematical Formulation**:
```
m₀(t) = 0
```
 
**Implementation Details**:
- Simply replaces all consumption values with zero
- When applied to monthly consumption data, all days in the month are set to zero
- Can be considered the most blatant form of energy theft

**Use Case**: This attack model represents a situation where a consumer completely bypasses the energy meter, resulting in no recorded consumption.

---

### Attack 1: Constant Factor Reduction

**Description**: Decreases consumption by a constant factor.

**Mathematical Formulation**:
```
m₁(t) = α × e(t)
```
where:
- e(t) is the original consumption at time t
- α is a constant random value between 0 and 1 (typically 0.1 to 0.8)

**Implementation Details**:
- Multiplies each reading by the same constant factor
- The factor is chosen randomly but remains constant for all readings
- For daily data: applies to each 30-minute reading
- For monthly data: applies the same factor to each daily reading

**Use Case**: Represents a situation where a consumer installs a device that consistently reduces the meter readings by a fixed percentage.

---

### Attack 2: Random Period Zero Consumption

**Description**: Sets consumption to zero for a random time period.

**Mathematical Formulation**:
```
m₂(t) = β(t) × e(t)
```
where:
- e(t) is the original consumption at time t
- β(t) is 0 between randomly chosen start time t₁ and end time t₂, and 1 otherwise

**Implementation Details**:
- For daily data: typically sets a 6-hour block to zero
- For monthly data: sets approximately one week of readings to zero
- The start time is chosen randomly

**Use Case**: Models theft where a consumer tamperers with the meter during specific time periods, such as during peak hours or when they know they'll consume significant amounts of energy.

---

### Attack 3: Variable Factor Reduction

**Description**: Decreases consumption by different random factors for each reading.

**Mathematical Formulation**:
```
m₃(t) = γ(t) × e(t)
```
where:
- e(t) is the original consumption at time t
- γ(t) is a random factor between 0 and 1 for each time t

**Implementation Details**:
- Unlike Attack 1, each reading gets a different random factor
- For daily data: each 30-minute interval has its own random factor
- For monthly data: each day has its own random factor
- Factors typically range from 0.1 to 0.8

**Use Case**: Represents a sophisticated form of theft where the reduction varies over time, potentially to avoid detection algorithms that look for consistent patterns.

---

### Attack 4: Time-Limited Reduction

**Description**: Decreases consumption by a constant factor during a random time period.

**Mathematical Formulation**:
```
m₄(t) = {
    α × e(t)  if t₁ < t < t₂
    e(t)      otherwise
}
```
where:
- e(t) is the original consumption at time t
- α is a constant factor between 0 and 1
- t₁ and t₂ define the start and end of the affected period

**Implementation Details**:
- Similar to Attack 2, but instead of setting values to zero, it reduces them by a factor
- For daily data: typically affects a 6-hour block
- For monthly data: affects approximately one week of readings

**Use Case**: Models theft that occurs during specific periods, such as during working hours when nobody is home to notice tampering.

---

### Attack 5: Mean-Based Substitution

**Description**: Replaces readings with random proportions of the mean consumption.

**Mathematical Formulation**:
```
m₅(t) = γ(t) × mean(e)
```
where:
- e is the original consumption time series
- mean(e) is the average consumption over the entire period
- γ(t) is a random factor between 0 and 1 for each time t

**Implementation Details**:
- Calculates the mean of all readings
- Replaces each reading with a random proportion of this mean
- Random factors typically range from 0.1 to 0.8
- Creates erratic behavior that still maintains some relationship to normal consumption

**Use Case**: Represents a form of theft where real consumption patterns are replaced with artificial ones that still maintain some statistical properties of the original.

---

### Attack 6: Upper-Bound Capping

**Description**: Caps consumption values at a predefined threshold.

**Mathematical Formulation**:
```
m₆(t) = {
    e(t)  if e(t) ≤ a
    a     if e(t) > a
}
```
where:
- e(t) is the original consumption at time t
- a is a threshold value (typically set as a proportion of maximum consumption)

**Implementation Details**:
- The threshold is typically set to 0.6 × max(e)
- Values below the threshold remain unchanged
- Values above the threshold are replaced with the threshold value
- Effectively removes consumption peaks

**Use Case**: Models a situation where a consumer installs a device that prevents the meter from recording consumption above a certain level, particularly targeting peak usage periods.

---

### Attack 7: Lower-Bound Reduction

**Description**: Subtracts a fixed amount from all readings, with a minimum of zero.

**Mathematical Formulation**:
```
m₇(t) = max(e(t) - a, 0)
```
where:
- e(t) is the original consumption at time t
- a is a constant value to subtract

**Implementation Details**:
- Typically sets a to a proportion of the maximum consumption (e.g., 0.6 × max(e))
- Ensures no negative values by using the max function
- Low consumption periods will become zero
- Higher consumption periods are reduced by a fixed amount

**Use Case**: Represents theft where a constant amount of energy is diverted around the meter, with the constraint that reported consumption cannot be negative.

---

### Attack 8: Progressive Reduction

**Description**: Gradually increases the intensity of theft over time.

**Mathematical Formulation**:
```
m₈(t) = (1 - i(t)) × e(t)
```
where:
- e(t) is the original consumption at time t
- i(t) is the attack intensity at time t, defined as:
  - 0 for t < t₁
  - s × (t - t₁) for t₁ ≤ t < t₂
  - i_max for t ≥ t₂
- s is the rate of change in intensity
- i_max is the maximum intensity

**Implementation Details**:
- Attack intensity starts at 0 and increases linearly until reaching maximum
- Typical values: i_max between 0.3 and 1.0, s between 0.05 and 0.1
- For daily data: progression happens over hours
- For monthly data: progression happens over days

**Use Case**: Models a situation where theft gradually increases over time, possibly to avoid sudden changes that might trigger detection systems.

---

### Attack 9: Average Substitution

**Description**: Replaces all readings with the average consumption value.

**Mathematical Formulation**:
```
m₉(t) = mean(e)
```
where:
- e is the original consumption time series
- mean(e) is the average consumption over the entire period

**Implementation Details**:
- Calculates the mean of all consumption readings
- Replaces every reading with this constant value
- For daily data: uses the daily average
- For monthly data: uses the monthly average

**Use Case**: Represents a form of theft where the meter is tampered to report a constant value based on typical usage, eliminating the natural variations in consumption patterns.

---

### Attack 10: Time Reversal

**Description**: Reverses the consumption pattern.

**Mathematical Formulation**:
```
m₁₀(t) = e(p - t)
```
where:
- e(t) is the original consumption at time t
- p is the total number of periods

**Implementation Details**:
- For daily data: reverses the order of the 48 half-hourly readings
- For monthly data: reverses the order of days within a week
- Preserves the total consumption but changes its temporal distribution

**Use Case**: Models a sophisticated form of theft that manipulates the timing of consumption to take advantage of time-of-use pricing, shifting apparent peak usage to off-peak times.

---

### Attack 11: Peak Reduction and Redistribution

**Description**: Reduces consumption during peak hours and redistributes the energy to other periods.

**Mathematical Formulation**:
```
m₁₁(t) = {
    e(t) - α × e(t)            if t₁ < t < t₂
    e(t) + ε / (N - n)         otherwise
}
```
where:
- e(t) is the original consumption at time t
- α is the reduction factor (typically 0.3)
- t₁ and t₂ define the peak period
- ε is the total energy reduced during peak hours
- N is the total number of time periods
- n is the number of time periods in the peak hours

**Implementation Details**:
- Identifies peak consumption period
- Reduces consumption during this period by a factor (typically 0.3)
- Redistributes the reduced energy equally across all other periods
- Preserves total consumption

**Use Case**: Models theft aimed at reducing high-cost peak consumption while maintaining the same total energy usage, suitable for systems with time-of-use pricing.

---

### Attack 12: Consumer Substitution

**Description**: Swaps the consumption pattern with that of a lower-consuming user.

**Mathematical Formulation**:
```
m₁₂(t) = e_low(t)
```
where:
- e_low(t) is the consumption of a different consumer with lower overall usage

**Implementation Details**:
- Identifies consumers with lower consumption (typically 40% or less)
- Replaces the target consumer's readings with those of the lower-consuming user
- For daily data: swaps the entire day's pattern
- For monthly data: swaps the entire month

**Use Case**: Represents sophisticated theft where a consumer's meter is programmed to report the consumption pattern of a different, lower-consuming household.

---

### Attack 13: Intermittent Zero Consumption

**Description**: Randomly switches between zero and actual consumption.

**Mathematical Formulation**:
```
m₁₃(t) = δ(t) × e(t)
```
where:
- e(t) is the original consumption at time t
- δ(t) is a random binary value (0 or 1) for each time t

**Implementation Details**:
- For each reading, randomly decides whether to report actual consumption or zero
- For daily data: makes this decision for each 30-minute interval
- For monthly data: makes this decision for each day

**Use Case**: Models intermittent theft or a malfunctioning meter that sometimes reports correct readings and sometimes reports zeros.

## Using Attack Models

All attack models are implemented in the `src/attack_models` directory. The base class `BaseAttackModel` provides a common interface, and each attack is implemented as a separate class that inherits from this base.

### Example: Applying an Attack Model

```python
from src.attack_models.implementations import ConstantFactorAttack

# Load data
data = load_data()

# Create the attack model
attack = ConstantFactorAttack()

# Apply the attack with a specific factor
attacked_data = attack.apply(data, alpha=0.5)
```

For more details on the implementation, see the source code in `src/attack_models/implementations.py`.

## References

1. Chuwa, Lucas & Maseng, Seemanta & Kenned, Joseph. (2021). Review of Energy Theft Detection in Smart Grid: Attack Models, Detection Techniques, and Open Issues. IEEE Access. 10. 10.1109/ACCESS.2021.3135805.

2. Jokar, P., Arianpoo, N., & Leung, V. C. M. (2016). Electricity Theft Detection in AMI Using Customers' Consumption Patterns. IEEE Transactions on Smart Grid, 7(1), 216-226.

3. Messinis, G. M., & Hatziargyriou, N. D. (2018). Review of non-technical loss detection methods. Electric Power Systems Research, 158, 250-266.
