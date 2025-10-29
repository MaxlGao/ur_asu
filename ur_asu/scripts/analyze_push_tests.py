import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_friction_trials(csv_file, t_start=1.0, t_end=5.0):
    """
    Process friction trial CSV files to extract velocity estimates.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    t_start : float
        Start time for clipping each trial (seconds)
    t_end : float
        End time for clipping each trial (seconds)
    
    Returns:
    --------
    results : pd.DataFrame
        DataFrame with trial results including velocities
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Identify trial boundaries (when Fy changes)
    df['trial'] = (df['Fy'] != df['Fy'].shift()).cumsum()
    
    # Reset time within each trial
    df['trial_time'] = df.groupby('trial')['time'].transform(lambda x: x - x.iloc[0])
    
    results = []
    
    for trial_id, trial_data in df.groupby('trial'):
        Fy = trial_data['Fy'].iloc[0]
        
        # Clip data to the specified time interval
        clipped = trial_data[(trial_data['trial_time'] >= t_start) & 
                             (trial_data['trial_time'] <= t_end)].copy()
        
        if len(clipped) < 2:
            print(f"Warning: Trial {trial_id} (Fy={Fy}) has insufficient data in interval [{t_start}, {t_end}]")
            continue
        
        # Calculate velocities using finite differences
        dt = np.diff(clipped['trial_time'])
        vx = np.diff(clipped['x']) / dt
        vy = np.diff(clipped['y']) / dt
        
        # Calculate speed
        speed = np.sqrt(vx**2 + vy**2)
        
        # Statistics for the trial
        result = {
            'trial': trial_id,
            'Fy': Fy,
            'n_points': len(clipped),
            'duration': clipped['trial_time'].iloc[-1] - clipped['trial_time'].iloc[0],
            'vx_mean': np.mean(vx),
            'vx_std': np.std(vx),
            'vy_mean': np.mean(vy),
            'vy_std': np.std(vy),
            'speed_mean': np.mean(speed),
            'speed_std': np.std(speed),
            'x_displacement': clipped['x'].iloc[-1] - clipped['x'].iloc[0],
            'y_displacement': clipped['y'].iloc[-1] - clipped['y'].iloc[0],
        }
        
        results.append(result)
    
    return pd.DataFrame(results)


def plot_trial_trajectories(csv_file, t_start=1.0, t_end=5.0):
    """
    Plot trajectories for each trial in the clipped interval.
    """
    df = pd.read_csv(csv_file)
    df['trial'] = (df['Fy'] != df['Fy'].shift()).cumsum()
    df['trial_time'] = df.groupby('trial')['time'].transform(lambda x: x - x.iloc[0])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Trajectories in x-y space
    for trial_id, trial_data in df.groupby('trial'):
        Fy = trial_data['Fy'].iloc[0]
        clipped = trial_data[(trial_data['trial_time'] >= t_start) & 
                             (trial_data['trial_time'] <= t_end)]
        
        if len(clipped) > 0:
            axes[0].plot(clipped['x'], clipped['y'], 
                        label=f'Fy={Fy}', marker='o', markersize=2, alpha=0.7)
    
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    axes[0].set_title('Trajectories (clipped interval)')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].axis('equal')
    
    # Plot 2: Position vs time
    for trial_id, trial_data in df.groupby('trial'):
        Fy = trial_data['Fy'].iloc[0]
        clipped = trial_data[(trial_data['trial_time'] >= t_start) & 
                             (trial_data['trial_time'] <= t_end)]
        
        if len(clipped) > 0:
            axes[1].plot(clipped['trial_time'], clipped['y'], 
                        label=f'Fy={Fy}', alpha=0.7)
    
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('y position (m)')
    axes[1].set_title('Y Position vs Time (clipped interval)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    return fig


def fit_velocity_force_relationship(results):
    """
    Fit a linear relationship: vy = a*Fy + b
    
    Parameters:
    -----------
    results : pd.DataFrame
        Results DataFrame from process_friction_trials
    
    Returns:
    --------
    a : float
        Slope of the linear fit
    b : float
        Intercept of the linear fit
    r_squared : float
        R-squared value of the fit
    """
    # Extract Fy and vy_mean
    Fy = results['Fy'].values
    vy = results['vy_mean'].values
    
    # Perform linear regression: vy = a*Fy + b
    coeffs = np.polyfit(Fy, vy, 1)
    a, b = coeffs[0], coeffs[1]
    
    # Calculate R-squared
    vy_pred = a * Fy + b
    ss_res = np.sum((vy - vy_pred)**2)
    ss_tot = np.sum((vy - np.mean(vy))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return a, b, r_squared


def plot_velocity_force_fit(results, a, b, r_squared):
    """
    Plot the velocity-force relationship and linear fit.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    Fy = results['Fy'].values
    vy = results['vy_mean'].values
    vy_std = results['vy_std'].values
    
    # Plot data points with error bars
    ax.errorbar(Fy, vy, yerr=vy_std, fmt='o', markersize=8, 
                capsize=5, label='Measured data', alpha=0.7)
    
    # Plot fitted line
    Fy_fit = np.linspace(Fy.min(), Fy.max(), 100)
    vy_fit = a * Fy_fit + b
    ax.plot(Fy_fit, vy_fit, 'r-', linewidth=2, 
            label=f'Fit: vy = {a:.6f}·Fy + {b:.6f}')
    
    ax.set_xlabel('Force Fy (N)', fontsize=12)
    ax.set_ylabel('Mean Velocity vy (m/s)', fontsize=12)
    ax.set_title(f'Velocity-Force Relationship (R² = {r_squared:.6f})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add text box with fit parameters
    textstr = f'a = {a:.6f} m/(s·N)\nb = {b:.6f} m/s\nR² = {r_squared:.6f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig

# Example usage
if __name__ == "__main__":
    # Process the trials
    # csvs = ['friction_trials_smooth_jenga.csv',
    #         'friction_trials_sticky_jenga.csv',
    #         'friction_trials_smooth_wrench.csv',
    #         'friction_trials_sticky_wrench.csv']
    csvs = ['~/ros2_ws/src/ur_asu/ur_asu/scripts/friction_trials_smooth_jenga_6_linear.csv']
    for csv in csvs:
        results = process_friction_trials(csv, 
                                        t_start=1.0, t_end=5.0)
        
        # Create summary statistics
        print(f"\nSummary by Force Level, for {csv}:")
        print("=" * 80)
        summary = results.groupby('Fy').agg({
            'vx_mean': 'mean',
            'vy_mean': 'mean',
            # 'speed_mean': 'mean',
            'x_displacement': 'mean',
            'y_displacement': 'mean'
        }).round(6)
        print(summary)

        # Perform linear fit
        a, b, r_squared = fit_velocity_force_relationship(results)
        print(f"\nLinear Fit: vy = {a:.6f} * Fy + {b:.6f}")
        print(f"R² = {r_squared:.6f}")
    
        # Plot trajectories
        fig = plot_trial_trajectories(csv, t_start=1.0, t_end=5.0)
        plt.show()