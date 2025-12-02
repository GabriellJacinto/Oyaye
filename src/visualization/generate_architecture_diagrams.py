#!/usr/bin/env python3
"""
Generate architecture visualization diagrams for NP-SNN model.
Creates comprehensive PNG diagrams showing model structure and data flow.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Arrow
import numpy as np
import os

# Set style for clean diagrams
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'

def create_overall_architecture_diagram():
    """Create high-level NP-SNN architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#E3F2FD',      # Light blue
        'encoder': '#BBDEFB',     # Blue
        'snn': '#2196F3',         # Deep blue
        'decoder': '#FFC107',     # Amber
        'output': '#4CAF50',      # Green
        'physics': '#FF5722',     # Deep orange
        'text': '#333333'
    }
    
    # Title
    ax.text(5, 9.5, 'Neural Physics-Informed Spiking Neural Network (NP-SNN)', 
            ha='center', va='center', fontsize=18, fontweight='bold', color=colors['text'])
    
    # Input layer
    input_box = FancyBboxPatch((0.5, 7.5), 1.5, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 8, 'Input\nObservations\n(6D states)', ha='center', va='center', fontweight='bold')
    
    # Observation Encoder
    encoder_box = FancyBboxPatch((2.5, 7.5), 1.5, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['encoder'],
                                 edgecolor='black', linewidth=2)
    ax.add_patch(encoder_box)
    ax.text(3.25, 8, 'Observation\nEncoder\n(2 layers)', ha='center', va='center', fontweight='bold')
    
    # Time Encoding
    time_box = FancyBboxPatch((4.5, 7.5), 1.5, 1,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['encoder'],
                              edgecolor='black', linewidth=2)
    ax.add_patch(time_box)
    ax.text(5.25, 8, 'Time Encoding\n(Fourier Features)\n64 dims', ha='center', va='center', fontweight='bold')
    
    # SNN Core (3 layers)
    for i, layer_name in enumerate(['SNN Layer 1', 'SNN Layer 2', 'SNN Layer 3']):
        y_pos = 5.5 - i * 1.2
        snn_box = FancyBboxPatch((2, y_pos), 2.5, 0.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['snn'],
                                 edgecolor='black', linewidth=2)
        ax.add_patch(snn_box)
        ax.text(3.25, y_pos + 0.4, f'{layer_name}\n256 LIF Neurons', 
                ha='center', va='center', fontweight='bold', color='white')
    
    # Ensemble Decoder
    decoder_box = FancyBboxPatch((6, 4.5), 2, 1.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['decoder'],
                                 edgecolor='black', linewidth=2)
    ax.add_patch(decoder_box)
    ax.text(7, 5.25, 'Ensemble Decoder\n5 Decoder Heads\nUncertainty Estimation', 
            ha='center', va='center', fontweight='bold')
    
    # Output
    output_box = FancyBboxPatch((6, 2.5), 2, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['output'],
                                edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(7, 3, 'Predicted States\n(6D orbital states)\n+ Uncertainty', 
            ha='center', va='center', fontweight='bold', color='white')
    
    # Physics-Informed Losses
    physics_box = FancyBboxPatch((0.5, 0.5), 7.5, 1.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['physics'],
                                 edgecolor='black', linewidth=2)
    ax.add_patch(physics_box)
    ax.text(4.25, 1.25, 'Physics-Informed Loss Components\n' +
            'Energy Conservation • Angular Momentum • Dynamics Residual • Measurement Consistency',
            ha='center', va='center', fontweight='bold', color='white', fontsize=12)
    
    # Arrows showing data flow
    # Input to encoder
    ax.arrow(2, 8, 0.4, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Encoder to time encoding
    ax.arrow(4, 8, 0.4, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Time encoding to SNN
    ax.arrow(5, 7.5, -1.5, -1.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Between SNN layers
    for i in range(2):
        y_start = 5.5 - i * 1.2
        ax.arrow(3.25, y_start, 0, -0.3, head_width=0.1, head_length=0.05, fc='black', ec='black')
    
    # SNN to decoder
    ax.arrow(4.5, 3.7, 1.4, 0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Decoder to output
    ax.arrow(7, 4.5, 0, -0.9, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Physics loss connections (dashed lines)
    for x_pos in [3.25, 7]:
        ax.plot([x_pos, x_pos], [2.8, 2], 'r--', linewidth=2, alpha=0.7)
    
    # Parameter count annotation
    ax.text(8.5, 9, 'Total Parameters:\n799,142', ha='center', va='center', 
            fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('docs/npsnn_architecture_overview.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Created: docs/npsnn_architecture_overview.png")


def create_snn_detailed_diagram():
    """Create detailed SNN layer structure diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'Spiking Neural Network Layer Details', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # LIF Neuron diagram
    neuron_x, neuron_y = 2, 5.5
    
    # Neuron body (circle)
    neuron = Circle((neuron_x, neuron_y), 0.3, facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(neuron)
    ax.text(neuron_x, neuron_y, 'LIF', ha='center', va='center', fontweight='bold')
    
    # Input spikes
    for i, spike_time in enumerate([1.2, 1.8, 2.4]):
        spike_x = neuron_x - 0.8
        spike_y = neuron_y + (i-1) * 0.2
        ax.plot([spike_x-0.2, spike_x], [spike_y, spike_y], 'g-', linewidth=3)
        ax.plot([spike_x, spike_x], [spike_y-0.05, spike_y+0.05], 'g-', linewidth=3)
        ax.text(spike_x-0.3, spike_y, f't={spike_time}', ha='right', va='center', fontsize=8)
    
    # Output spike
    output_x = neuron_x + 0.8
    ax.plot([output_x, output_x+0.2], [neuron_y, neuron_y], 'r-', linewidth=3)
    ax.plot([output_x+0.1, output_x+0.1], [neuron_y-0.05, neuron_y+0.05], 'r-', linewidth=3)
    ax.text(output_x+0.3, neuron_y, 'Spike Output', ha='left', va='center', fontweight='bold')
    
    # Membrane potential plot
    ax.text(2, 4.5, 'Membrane Potential Dynamics', ha='center', va='center', fontweight='bold', fontsize=12)
    
    # Create small subplot for membrane potential
    t = np.linspace(0, 5, 1000)
    v_mem = np.zeros_like(t)
    threshold = 0.5
    decay = 0.9
    
    # Simulate simple LIF dynamics
    v = 0
    spike_times = [1.2, 1.8, 2.4, 3.5]
    for i, time in enumerate(t):
        if any(abs(time - st) < 0.01 for st in spike_times):
            v += 0.3  # Input spike
        if v >= threshold:
            v = 0  # Reset after spike
        else:
            v *= 0.999  # Decay
        v_mem[i] = v
    
    # Plot membrane potential in a box
    mem_box = Rectangle((0.5, 3), 3, 1.2, facecolor='white', edgecolor='black')
    ax.add_patch(mem_box)
    
    # Normalize for plotting in the box
    t_norm = 0.5 + (t / 5) * 3
    v_norm = 3 + (v_mem / 0.6) * 1.2
    ax.plot(t_norm, v_norm, 'b-', linewidth=2)
    
    # Threshold line
    thresh_norm = 3 + (threshold / 0.6) * 1.2
    ax.axhline(y=thresh_norm, xmin=0.05, xmax=0.35, color='r', linestyle='--', linewidth=2)
    ax.text(3.6, thresh_norm, 'Threshold = 0.5', ha='left', va='center', color='red', fontweight='bold')
    
    ax.text(0.3, 3.6, 'V', ha='center', va='center', fontweight='bold', rotation=90)
    ax.text(2, 2.8, 'Time', ha='center', va='center', fontweight='bold')
    
    # Layer structure
    ax.text(6.5, 6.5, 'SNN Layer Structure (256 neurons)', ha='center', va='center', 
            fontsize=14, fontweight='bold')
    
    # Draw grid of neurons
    for row in range(8):
        for col in range(8):
            x_pos = 5.5 + col * 0.2
            y_pos = 6 - row * 0.15
            if row < 4:  # First half in blue
                color = 'lightblue'
            else:  # Second half in lightgreen
                color = 'lightgreen'
            neuron_small = Circle((x_pos, y_pos), 0.05, facecolor=color, edgecolor='gray')
            ax.add_patch(neuron_small)
    
    # Add "..." to indicate more neurons
    ax.text(7.2, 5, '...', ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(6.5, 4.5, '256 LIF Neurons Total', ha='center', va='center', fontweight='bold')
    
    # Parameters box
    param_text = """SNN Layer Parameters:
    • Neurons: 256 LIF neurons
    • Membrane Threshold: 0.5
    • Membrane Decay: 0.9
    • Dropout Rate: 0.1
    • Time Steps: Variable"""
    
    ax.text(6.5, 2.5, param_text, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('docs/snn_layer_details.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Created: docs/snn_layer_details.png")


def create_training_flow_diagram():
    """Create curriculum learning training flow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'NP-SNN Curriculum Learning Training Flow', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Training stages
    stages = [
        {'name': 'Sanity\nStage', 'epochs': '0-4', 'color': '#E8F5E8', 'desc': 'Basic\nValidation'},
        {'name': 'Supervised\nStage', 'epochs': '5-54', 'color': '#FFF3E0', 'desc': 'Trajectory\nFitting'},
        {'name': 'Mixed\nStage', 'epochs': '55-134', 'color': '#E3F2FD', 'desc': 'Balanced\nTraining'},
        {'name': 'Physics\nStage', 'epochs': '135-184', 'color': '#FCE4EC', 'desc': 'Physics\nDominated'},
        {'name': 'Fine-tune\nStage', 'epochs': '185-199', 'color': '#F3E5F5', 'desc': 'Final\nOptimization'}
    ]
    
    # Draw stages as connected boxes
    box_width = 2
    box_height = 1.2
    start_x = 1
    y_pos = 5.5
    
    for i, stage in enumerate(stages):
        x_pos = start_x + i * 2.2
        
        # Stage box
        stage_box = FancyBboxPatch((x_pos, y_pos), box_width, box_height,
                                   boxstyle="round,pad=0.1",
                                   facecolor=stage['color'],
                                   edgecolor='black', linewidth=2)
        ax.add_patch(stage_box)
        
        # Stage text
        ax.text(x_pos + box_width/2, y_pos + box_height/2 + 0.2, stage['name'],
                ha='center', va='center', fontweight='bold', fontsize=10)
        ax.text(x_pos + box_width/2, y_pos + box_height/2 - 0.1, f"Epochs {stage['epochs']}",
                ha='center', va='center', fontsize=8)
        ax.text(x_pos + box_width/2, y_pos + box_height/2 - 0.3, stage['desc'],
                ha='center', va='center', fontsize=8, style='italic')
        
        # Arrow to next stage
        if i < len(stages) - 1:
            arrow_start = x_pos + box_width
            arrow_end = x_pos + box_width + 0.2
            ax.arrow(arrow_start, y_pos + box_height/2, 0.2, 0,
                    head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Progress indicator - show we completed stages 1-2
    completed_x = start_x + 2 * 2.2 + box_width  # After stage 2
    ax.axvline(x=completed_x, ymin=0.6, ymax=0.8, color='green', linewidth=4)
    ax.text(completed_x, 4.8, 'Current\nProgress', ha='center', va='center', 
            fontweight='bold', color='green',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.7))
    
    # Loss components evolution
    ax.text(6, 4, 'Loss Component Evolution Throughout Training', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Create loss evolution diagram
    loss_components = ['Supervised Loss', 'Energy Conservation', 'Momentum Conservation', 
                      'Dynamics Residual', 'Measurement Loss']
    colors_loss = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    # Draw loss weight evolution bars
    for i, (component, color) in enumerate(zip(loss_components, colors_loss)):
        y_comp = 3 - i * 0.4
        
        # Component name
        ax.text(0.5, y_comp, component, ha='right', va='center', fontweight='bold')
        
        # Weight bars for each stage (simplified representation)
        weights = [
            [1.0, 0.1, 0.1, 0.1, 0.1],  # Supervised dominates early
            [0.8, 0.2, 0.2, 0.2, 0.2],  # Supervised stage
            [0.5, 0.5, 0.5, 0.5, 0.5],  # Mixed stage
            [0.2, 0.8, 0.8, 0.8, 0.3],  # Physics stage
            [0.1, 0.9, 0.9, 0.9, 0.2]   # Fine-tune stage
        ]
        
        for j in range(5):  # 5 stages
            weight = weights[i][j]
            bar_x = 1 + j * 2.2
            bar_width = weight * 1.5
            
            bar = Rectangle((bar_x, y_comp - 0.1), bar_width, 0.2,
                           facecolor=color, alpha=0.7, edgecolor='black')
            ax.add_patch(bar)
    
    # Legend
    ax.text(11, 3, 'Weight\nIntensity', ha='center', va='center', fontweight='bold')
    ax.text(11, 2.5, 'High', ha='center', va='center')
    high_bar = Rectangle((10.7, 2.3), 0.6, 0.1, facecolor='gray', alpha=0.8)
    ax.add_patch(high_bar)
    ax.text(11, 2.1, 'Low', ha='center', va='center')
    low_bar = Rectangle((10.7, 1.9), 0.2, 0.1, facecolor='gray', alpha=0.3)
    ax.add_patch(low_bar)
    
    plt.tight_layout()
    plt.savefig('docs/training_flow_curriculum.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Created: docs/training_flow_curriculum.png")


def create_data_flow_diagram():
    """Create data processing and normalization flow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'Data Processing and Normalization Pipeline', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Data flow steps
    steps = [
        {'name': 'Raw Orbital\nTrajectories', 'pos': (1, 6), 'color': '#FFEBEE'},
        {'name': 'Scenario\nGeneration', 'pos': (3, 6), 'color': '#E8F5E8'},
        {'name': 'Observation\nSimulation', 'pos': (5, 6), 'color': '#E3F2FD'},
        {'name': 'Data\nNormalization', 'pos': (7, 6), 'color': '#FFF3E0'},
        {'name': 'Training\nBatches', 'pos': (9, 6), 'color': '#F3E5F5'}
    ]
    
    # Draw processing steps
    for i, step in enumerate(steps):
        x, y = step['pos']
        
        # Step box
        step_box = FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.8,
                                  boxstyle="round,pad=0.1",
                                  facecolor=step['color'],
                                  edgecolor='black', linewidth=2)
        ax.add_patch(step_box)
        ax.text(x, y, step['name'], ha='center', va='center', fontweight='bold')
        
        # Arrow to next step
        if i < len(steps) - 1:
            ax.arrow(x + 0.6, y, 0.8, 0, head_width=0.1, head_length=0.1, 
                    fc='black', ec='black')
    
    # Detailed normalization section
    ax.text(5, 4.5, 'Critical Normalization Transform', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='red')
    
    # Before normalization
    before_box = FancyBboxPatch((1, 3), 3.5, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor='#FFCDD2',
                                edgecolor='red', linewidth=2)
    ax.add_patch(before_box)
    
    before_text = """BEFORE (Unstable):
    Position: ~6.7×10⁶ m
    Velocity: ~7.5×10³ m/s
    Loss: 4.5×10¹³ (Divergent)"""
    ax.text(2.75, 3.6, before_text, ha='center', va='center', fontsize=10, fontfamily='monospace')
    
    # Normalization arrow
    ax.arrow(4.5, 3.6, 1, 0, head_width=0.15, head_length=0.15, 
            fc='red', ec='red', linewidth=3)
    ax.text(5, 3.3, 'Normalize', ha='center', va='center', fontweight='bold', color='red')
    
    # After normalization  
    after_box = FancyBboxPatch((5.5, 3), 3.5, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor='#C8E6C9',
                               edgecolor='green', linewidth=2)
    ax.add_patch(after_box)
    
    after_text = """AFTER (Stable):
    Position: ~0.67 (÷10⁷)
    Velocity: ~0.75 (÷10⁴) 
    Loss: 0.531 (Converged)"""
    ax.text(7.25, 3.6, after_text, ha='center', va='center', fontsize=10, fontfamily='monospace')
    
    # Normalization formulas
    formula_box = FancyBboxPatch((2, 1.5), 6, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor='lightyellow',
                                 edgecolor='orange', linewidth=2)
    ax.add_patch(formula_box)
    
    formula_text = """Normalization Transforms:
    position_norm = position / 1×10⁷     velocity_norm = velocity / 1×10⁴"""
    ax.text(5, 2, formula_text, ha='center', va='center', fontsize=12, 
            fontfamily='monospace', fontweight='bold')
    
    # Impact annotation
    impact_text = """10¹³× Loss Reduction
    Training Stability Achieved"""
    ax.text(5, 0.7, impact_text, ha='center', va='center', fontsize=14, 
            fontweight='bold', color='green',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('docs/data_normalization_flow.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Created: docs/data_normalization_flow.png")


def main():
    """Generate all architecture diagrams."""
    print("🎨 Generating NP-SNN Architecture Visualizations...")
    
    # Ensure docs directory exists
    os.makedirs('docs', exist_ok=True)
    
    # Generate all diagrams
    create_overall_architecture_diagram()
    create_snn_detailed_diagram()
    create_training_flow_diagram()
    create_data_flow_diagram()
    
    print("\n✅ All architecture diagrams generated successfully!")
    print("\nGenerated files:")
    print("📊 docs/npsnn_architecture_overview.png - Complete system architecture")
    print("🧠 docs/snn_layer_details.png - Detailed SNN layer structure") 
    print("📈 docs/training_flow_curriculum.png - Curriculum learning progression")
    print("🔄 docs/data_normalization_flow.png - Data processing pipeline")
    
    print(f"\n💡 Add these images to your experimental_results.md using:")
    print("![NP-SNN Architecture](npsnn_architecture_overview.png)")
    print("![SNN Details](snn_layer_details.png)")
    print("![Training Flow](training_flow_curriculum.png)")
    print("![Data Pipeline](data_normalization_flow.png)")

if __name__ == "__main__":
    main()