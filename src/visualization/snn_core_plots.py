"""
SNN Core Visualization Suite

This module creates comprehensive visualizations for the Neural Physics-Informed
Spiking Neural Network core architecture, including:

1. Network Architecture Diagrams
2. LIF Neuron Dynamics 
3. Spike Pattern Analysis
4. Membrane Potential Evolution
5. Multi-timescale Behavior
6. Gradient Flow Visualization
7. Physics Compatibility Analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

from src.models.snn_core import create_physics_snn, create_standard_snn, SNNConfig, AdaptiveLIFLayer, MultiTimescaleLIFLayer
from src.models.time_encoding import MixedTimeEncoding

class SNNVisualizationSuite:
    """Comprehensive visualization suite for SNN core analysis."""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        self.colors = {
            'membrane': '#FF6B6B',
            'spikes': '#4ECDC4', 
            'threshold': '#45B7D1',
            'input': '#96CEB4',
            'fast': '#FFEAA7',
            'slow': '#DDA0DD',
            'residual': '#FFB6C1'
        }
    
    def visualize_architecture(self, snn_config: SNNConfig = None):
        """Visualize SNN architecture diagram."""
        if snn_config is None:
            snn_config = SNNConfig(
                input_size=32,
                hidden_sizes=[64, 128, 64],
                output_size=6,
                multi_timescale=True,
                residual=True
            )
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Network layers
        layers = [snn_config.input_size] + snn_config.hidden_sizes + [snn_config.output_size]
        layer_names = ['Time\nEncoded\nInput'] + [f'Hidden\n{i+1}' for i in range(len(snn_config.hidden_sizes))] + ['Orbital\nState\nOutput']
        
        # Position layers
        x_positions = np.linspace(0.1, 0.9, len(layers))
        y_center = 0.5
        
        # Draw layers
        layer_boxes = []
        for i, (size, name, x_pos) in enumerate(zip(layers, layer_names, x_positions)):
            # Calculate box height based on layer size
            box_height = min(0.6, max(0.1, size / max(layers) * 0.5))
            box_width = 0.12
            
            # Special styling for different layer types
            if i == 0:
                # Input layer
                color = self.colors['input']
                alpha = 0.7
            elif i == len(layers) - 1:
                # Output layer  
                color = self.colors['membrane']
                alpha = 0.7
            else:
                # Hidden layers
                if i == 1 and snn_config.multi_timescale:
                    # Multi-timescale layer
                    color = self.colors['fast']
                    alpha = 0.8
                else:
                    color = self.colors['spikes']
                    alpha = 0.7
            
            # Draw layer box
            box = FancyBboxPatch(
                (x_pos - box_width/2, y_center - box_height/2),
                box_width, box_height,
                boxstyle="round,pad=0.01",
                facecolor=color,
                edgecolor='black',
                alpha=alpha,
                linewidth=2
            )
            ax.add_patch(box)
            layer_boxes.append((x_pos, y_center, box_width, box_height))
            
            # Add layer info
            ax.text(x_pos, y_center + box_height/2 + 0.08, name, 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.text(x_pos, y_center, f'{size}', 
                   ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        # Draw connections
        for i in range(len(layer_boxes) - 1):
            x1, y1, w1, h1 = layer_boxes[i]
            x2, y2, w2, h2 = layer_boxes[i + 1]
            
            # Forward connections
            arrow = ConnectionPatch(
                (x1 + w1/2, y1), (x2 - w2/2, y2),
                "data", "data",
                arrowstyle="->", shrinkA=5, shrinkB=5,
                mutation_scale=20, fc="black", alpha=0.7, linewidth=2
            )
            ax.add_patch(arrow)
        
        # Draw residual connections if enabled
        if snn_config.residual:
            x1, y1, w1, h1 = layer_boxes[0]  # Input
            x2, y2, w2, h2 = layer_boxes[1]  # First hidden
            
            # Curved residual connection
            arrow = ConnectionPatch(
                (x1 + w1/2, y1 + h1/4), (x2 - w2/2, y2 + h2/4),
                "data", "data",
                arrowstyle="->", shrinkA=5, shrinkB=5,
                connectionstyle="arc3,rad=0.3",
                mutation_scale=15, fc=self.colors['residual'], 
                ec=self.colors['residual'], alpha=0.6, linewidth=2
            )
            ax.add_patch(arrow)
            
            ax.text(x1 + (x2-x1)/2, y1 + 0.25, 'Residual\nConnection', 
                   ha='center', va='center', fontsize=9, 
                   color=self.colors['residual'], fontweight='bold')
        
        # Add multi-timescale annotation
        if snn_config.multi_timescale:
            x1, y1, w1, h1 = layer_boxes[1]
            ax.text(x1, y1 - h1/2 - 0.15, 'Multi-Timescale\nLIF Dynamics', 
                   ha='center', va='top', fontsize=9, 
                   color=self.colors['fast'], fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.title('Neural Physics-Informed SNN Architecture', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        legend_elements = [
            patches.Patch(color=self.colors['input'], label='Input Layer'),
            patches.Patch(color=self.colors['fast'], label='Multi-Timescale Layer'),
            patches.Patch(color=self.colors['spikes'], label='Standard LIF Layer'),
            patches.Patch(color=self.colors['membrane'], label='Output Layer'),
            patches.Patch(color=self.colors['residual'], label='Residual Connection')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        return fig
    
    def visualize_lif_dynamics(self, time_steps=200, dt=0.001):
        """Visualize LIF neuron dynamics and behavior."""
        
        # Create LIF layer for testing
        lif_layer = AdaptiveLIFLayer(
            input_size=10, 
            hidden_size=1, 
            beta=0.9,
            threshold=1.0,
            adaptive_beta=True
        )
        
        # Generate input current (step function + noise)
        times = np.arange(time_steps) * dt
        input_current = torch.zeros(1, 10)
        
        # Step inputs at different times
        step_times = [50, 100, 150]
        step_magnitudes = [0.8, 1.5, 0.6]
        
        membrane_potentials = []
        spikes = []
        input_currents = []
        
        mem = None
        
        for t in range(time_steps):
            # Generate input
            current_input = torch.zeros(1, 10)
            for step_time, magnitude in zip(step_times, step_magnitudes):
                if t >= step_time and t < step_time + 30:
                    current_input[0, :3] = magnitude
            
            # Add some noise
            current_input += 0.1 * torch.randn_like(current_input)
            
            # Forward pass
            with torch.no_grad():
                spike_out, mem = lif_layer(current_input, mem)
            
            # Store results
            membrane_potentials.append(mem[0, 0].item())
            spikes.append(spike_out[0, 0].item())
            input_currents.append(current_input[0, :3].mean().item())
        
        # Create visualization
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Input current
        axes[0].plot(times * 1000, input_currents, color=self.colors['input'], linewidth=2)
        axes[0].set_ylabel('Input Current', fontweight='bold')
        axes[0].set_title('LIF Neuron Dynamics Simulation', fontweight='bold', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 2: Membrane potential
        axes[1].plot(times * 1000, membrane_potentials, color=self.colors['membrane'], linewidth=2)
        axes[1].axhline(y=1.0, color=self.colors['threshold'], linestyle='--', linewidth=2, label='Threshold')
        axes[1].set_ylabel('Membrane Potential (V)', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Plot 3: Spikes
        spike_times = [i * dt * 1000 for i, spike in enumerate(spikes) if spike > 0.5]
        axes[2].eventplot(spike_times, colors=[self.colors['spikes']], linewidths=3, linelengths=0.8)
        axes[2].set_ylabel('Spikes', fontweight='bold')
        axes[2].set_xlabel('Time (ms)', fontweight='bold')
        axes[2].set_ylim(-0.5, 1.5)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def visualize_multitimescale_dynamics(self, time_steps=300):
        """Visualize multi-timescale LIF behavior."""
        
        # Create multi-timescale layer
        multi_lif = MultiTimescaleLIFLayer(
            input_size=5,
            hidden_size=2,
            beta_fast=0.8,
            beta_slow=0.99
        )
        
        # Generate oscillatory input (orbital-like)
        times = np.arange(time_steps)
        
        # Simulate orbital period effects
        fast_freq = 2 * np.pi / 50  # Fast dynamics (attitude)
        slow_freq = 2 * np.pi / 200  # Slow dynamics (orbital)
        
        mem_fast_history = []
        mem_slow_history = []
        spike_history = []
        input_history = []
        
        state = None
        
        for t in times:
            # Generate composite input
            fast_component = 0.3 * np.sin(fast_freq * t)
            slow_component = 0.5 * np.sin(slow_freq * t) 
            noise = 0.1 * np.random.randn()
            
            input_val = fast_component + slow_component + noise + 0.2
            input_tensor = torch.tensor([[input_val] * 5], dtype=torch.float32)
            
            # Forward pass
            with torch.no_grad():
                spikes, state = multi_lif(input_tensor, state)
            
            # Store results (first neuron)
            mem_fast_history.append(state['mem_fast'][0, 0].item())
            mem_slow_history.append(state['mem_slow'][0, 0].item()) 
            spike_history.append(spikes[0, 0].item())
            input_history.append(input_val)
        
        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # Plot 1: Input signal
        axes[0].plot(times, input_history, color=self.colors['input'], linewidth=2, label='Composite Input')
        axes[0].plot(times, [0.3 * np.sin(fast_freq * t) for t in times], 
                    color=self.colors['fast'], alpha=0.6, linewidth=1, label='Fast Component')
        axes[0].plot(times, [0.5 * np.sin(slow_freq * t) for t in times], 
                    color=self.colors['slow'], alpha=0.6, linewidth=1, label='Slow Component')
        axes[0].set_ylabel('Input Current', fontweight='bold')
        axes[0].set_title('Multi-Timescale LIF Dynamics', fontweight='bold', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Fast membrane dynamics
        axes[1].plot(times, mem_fast_history, color=self.colors['fast'], linewidth=2, label='Fast Membrane')
        axes[1].axhline(y=1.0, color=self.colors['threshold'], linestyle='--', alpha=0.7, label='Threshold')
        axes[1].set_ylabel('Fast Membrane (V)', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Slow membrane dynamics  
        axes[2].plot(times, mem_slow_history, color=self.colors['slow'], linewidth=2, label='Slow Membrane')
        axes[2].axhline(y=1.0, color=self.colors['threshold'], linestyle='--', alpha=0.7, label='Threshold')
        axes[2].set_ylabel('Slow Membrane (V)', fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Spike output
        spike_times = [t for t, spike in enumerate(spike_history) if spike > 0.5]
        if spike_times:
            axes[3].eventplot(spike_times, colors=[self.colors['spikes']], linewidths=3, linelengths=0.8)
        axes[3].set_ylabel('Spikes', fontweight='bold')
        axes[3].set_xlabel('Time Steps', fontweight='bold')
        axes[3].set_ylim(-0.5, 1.5)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def visualize_network_response(self, sequence_length=100):
        """Visualize full network response to time-encoded orbital data."""
        
        # Create network and time encoder
        time_encoder = MixedTimeEncoding(d_model=32, fourier_dim=16, learned_dim=16)
        snn = create_physics_snn(input_size=32, hidden_sizes=[64, 128, 64], output_size=6)
        
        # Generate orbital-like time sequence
        times = torch.linspace(0, 200, sequence_length).unsqueeze(0)  # One batch
        
        # Encode times
        with torch.no_grad():
            encoded_times = time_encoder(times)  # (1, seq_len, 32)
        
        # Process through SNN
        hidden_states = snn.init_hidden_states(1, 'cpu')
        outputs = []
        membrane_stats = []
        
        for t in range(sequence_length):
            with torch.no_grad():
                output, hidden_states = snn(encoded_times[:, t, :], hidden_states)
                outputs.append(output[0].numpy())  # Remove batch dim
                
                # Get membrane stats
                stats = snn.get_membrane_stats(hidden_states)
                membrane_stats.append(stats)
        
        outputs = np.array(outputs)  # (seq_len, 6)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Time encoding visualization
        ax1 = fig.add_subplot(gs[0, :])
        im = ax1.imshow(encoded_times[0].T.numpy(), aspect='auto', cmap='viridis', alpha=0.8)
        ax1.set_title('Time Encoding Features', fontweight='bold')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Feature Dimension')
        plt.colorbar(im, ax=ax1, shrink=0.6)
        
        # Plot 2-4: Orbital state outputs
        state_names = ['X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity']
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
        
        ax2 = fig.add_subplot(gs[1, :2])
        for i in range(3):  # Position components
            ax2.plot(outputs[:, i], color=colors[i], linewidth=2, label=state_names[i])
        ax2.set_title('Position Estimates', fontweight='bold')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Position Output')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[1, 2])
        for i in range(3, 6):  # Velocity components
            ax3.plot(outputs[:, i], color=colors[i], linewidth=2, label=state_names[i])
        ax3.set_title('Velocity Estimates', fontweight='bold')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Velocity Output')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 5: Membrane potential evolution
        ax4 = fig.add_subplot(gs[2, 0])
        mem_means = [stats.get('layer_1_mem_mean', 0) for stats in membrane_stats]
        mem_stds = [stats.get('layer_1_mem_std', 0) for stats in membrane_stats]
        
        ax4.plot(mem_means, color=self.colors['membrane'], linewidth=2, label='Mean')
        ax4.fill_between(range(len(mem_means)), 
                        np.array(mem_means) - np.array(mem_stds),
                        np.array(mem_means) + np.array(mem_stds),
                        alpha=0.3, color=self.colors['membrane'])
        ax4.set_title('Membrane Dynamics', fontweight='bold')
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Membrane Potential')
        ax4.grid(True, alpha=0.3)
        
        # Plot 6: Spike rates
        ax5 = fig.add_subplot(gs[2, 1])
        spike_rates = [stats.get('layer_1_spike_rate', 0) for stats in membrane_stats]
        ax5.plot(spike_rates, color=self.colors['spikes'], linewidth=2)
        ax5.set_title('Spike Rate Evolution', fontweight='bold')
        ax5.set_xlabel('Time Steps')
        ax5.set_ylabel('Spike Rate')
        ax5.grid(True, alpha=0.3)
        
        # Plot 7: Output magnitude
        ax6 = fig.add_subplot(gs[2, 2])
        output_norms = np.linalg.norm(outputs, axis=1)
        ax6.plot(output_norms, color='black', linewidth=2)
        ax6.set_title('Output Magnitude', fontweight='bold')
        ax6.set_xlabel('Time Steps')
        ax6.set_ylabel('||Output||')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('SNN Network Response Analysis', fontsize=16, fontweight='bold')
        
        return fig
    
    def visualize_physics_compatibility(self):
        """Visualize physics-related properties of the SNN."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Membrane time constants across layers
        betas = np.array([0.7, 0.8, 0.9, 0.95, 0.99])
        time_constants = -1 / np.log(betas)  # Effective time constant
        
        axes[0, 0].bar(range(len(betas)), time_constants, 
                      color=self.colors['membrane'], alpha=0.7)
        axes[0, 0].set_title('Membrane Time Constants', fontweight='bold')
        axes[0, 0].set_xlabel('Beta Value')
        axes[0, 0].set_ylabel('Time Constant')
        axes[0, 0].set_xticks(range(len(betas)))
        axes[0, 0].set_xticklabels([f'{b:.2f}' for b in betas])
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Spike frequency vs input current
        currents = np.linspace(0, 3, 50)
        spike_rates = []
        
        for current in currents:
            # Simple LIF rate approximation
            if current <= 1.0:
                rate = 0
            else:
                rate = 1 / (-np.log(1 - 1/current) + 1e-6)
            spike_rates.append(min(rate, 100))  # Cap at 100 Hz
        
        axes[0, 1].plot(currents, spike_rates, color=self.colors['spikes'], linewidth=3)
        axes[0, 1].axvline(x=1.0, color=self.colors['threshold'], 
                          linestyle='--', label='Threshold')
        axes[0, 1].set_title('Input-Output Relationship', fontweight='bold')
        axes[0, 1].set_xlabel('Input Current')
        axes[0, 1].set_ylabel('Spike Rate (Hz)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Multi-timescale separation
        fast_freqs = np.logspace(-2, 2, 100)
        fast_response = 1 / (1 + (fast_freqs / 10)**2)
        slow_response = 1 / (1 + (fast_freqs / 0.1)**2)
        
        axes[1, 0].semilogx(fast_freqs, fast_response, 
                           color=self.colors['fast'], linewidth=3, label='Fast Dynamics')
        axes[1, 0].semilogx(fast_freqs, slow_response,
                           color=self.colors['slow'], linewidth=3, label='Slow Dynamics')
        axes[1, 0].set_title('Frequency Response', fontweight='bold')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Response Amplitude')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Energy considerations
        orbital_periods = np.array([90, 120, 180, 360, 720])  # minutes
        encoding_freqs = 1 / (orbital_periods * 60)  # Hz
        energy_scales = orbital_periods**(-1.5)  # Simplified orbital energy scaling
        
        axes[1, 1].scatter(encoding_freqs * 1000, energy_scales, 
                          s=100, color=self.colors['input'], alpha=0.7)
        axes[1, 1].set_xlabel('Encoding Frequency (mHz)')
        axes[1, 1].set_ylabel('Relative Energy Scale')
        axes[1, 1].set_title('Orbital Frequency-Energy Relationship', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add period labels
        for i, period in enumerate(orbital_periods):
            axes[1, 1].annotate(f'{period}min', 
                               (encoding_freqs[i] * 1000, energy_scales[i]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=9)
        
        plt.suptitle('Physics Compatibility Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def generate_all_visualizations(self, save_path='visualizations/snn_core/'):
        """Generate and save all visualizations."""
        
        # Create directory
        os.makedirs(save_path, exist_ok=True)
        
        print("🎨 Generating SNN Core Visualizations...")
        
        # 1. Architecture diagram
        print("  → Network architecture diagram")
        fig1 = self.visualize_architecture()
        fig1.savefig(f'{save_path}snn_architecture.png', dpi=300, bbox_inches='tight')
        
        # 2. LIF dynamics
        print("  → LIF neuron dynamics")
        fig2 = self.visualize_lif_dynamics()
        fig2.savefig(f'{save_path}lif_dynamics.png', dpi=300, bbox_inches='tight')
        
        # 3. Multi-timescale dynamics
        print("  → Multi-timescale behavior")
        fig3 = self.visualize_multitimescale_dynamics()
        fig3.savefig(f'{save_path}multitimescale_dynamics.png', dpi=300, bbox_inches='tight')
        
        # 4. Network response
        print("  → Full network response analysis")
        fig4 = self.visualize_network_response()
        fig4.savefig(f'{save_path}network_response.png', dpi=300, bbox_inches='tight')
        
        # 5. Physics compatibility
        print("  → Physics compatibility analysis")
        fig5 = self.visualize_physics_compatibility()
        fig5.savefig(f'{save_path}physics_compatibility.png', dpi=300, bbox_inches='tight')
        
        plt.close('all')
        
        print(f"✅ All visualizations saved to {save_path}")
        print("\nGenerated files:")
        print(f"  • {save_path}snn_architecture.png")
        print(f"  • {save_path}lif_dynamics.png") 
        print(f"  • {save_path}multitimescale_dynamics.png")
        print(f"  • {save_path}network_response.png")
        print(f"  • {save_path}physics_compatibility.png")


def main():
    """Run visualization suite."""
    suite = SNNVisualizationSuite()
    
    # Generate all visualizations
    suite.generate_all_visualizations()
    
    # Also display one interactively as example
    print("\n🖼️  Displaying architecture diagram...")
    fig = suite.visualize_architecture()
    plt.show()


if __name__ == "__main__":
    main()