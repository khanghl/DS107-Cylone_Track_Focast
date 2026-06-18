import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# Create plots directory if it doesn't exist
os.makedirs("../experiments/results/plots", exist_ok=True)

df = pd.read_csv("../experiments/detailed_trajectories_predictions.csv")

# Danh sách 2 cơn bão cần vẽ
storms_to_plot = {
    "2024244N09137": "YAGI",
    "2025260N13138": "RAGASA"
}

def smooth(x, y):
    if len(x) < 3:
        return x, y
    t = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, 100)
    return np.interp(t_new, t, x), np.interp(t_new, t, y)

horizons = ['+6h', '+12h', '+24h']

for sid, storm_name in storms_to_plot.items():
    print(f"Plotting for storm: {storm_name} ({sid})")
    
    storm_df = df[df["SID"] == sid].copy()
    if storm_df.empty:
        print(f"Warning: No data found for {sid}")
        continue

    storm_df["time"] = pd.to_datetime(
        storm_df["YEAR"].astype(str) + "-" +
        storm_df["MONTH"].astype(str) + "-" +
        storm_df["DAY"].astype(str) + " " +
        storm_df["HOUR"].astype(str) + ":00"
    )
    storm_df = storm_df.sort_values("time")

    real_df = storm_df.drop_duplicates(subset=["time"])
    real_lon = real_df["BASE_LON"].values
    real_lat = real_df["BASE_LAT"].values
    real_lon_s, real_lat_s = smooth(real_lon, real_lat)

    # Lọc các model tốt nhất để vẽ cho đỡ rối mắt
    plot_models = [m for m in storm_df["Strategy"].unique() 
                   if "CLIPER" in m or "LIGHTGBM" in m or "Ensemble" in m or "Stacking" in m]
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_models)))

    # ========================================================
    # VẼ BẢN ĐỒ QUỸ ĐẠO (TRAJECTORIES)
    # ========================================================
    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    for ax, horizon in zip(axes, horizons):
        # Thiết lập bản đồ
        ax.add_feature(cfeature.LAND, facecolor="#f2f2f2")
        ax.add_feature(cfeature.OCEAN, facecolor="#d6ecff")
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        
        # Lấy dynamic extent dựa trên tọa độ bão
        min_lon, max_lon = real_lon.min() - 5, real_lon.max() + 5
        min_lat, max_lat = real_lat.min() - 5, real_lat.max() + 5
        ax.set_extent([min_lon, max_lon, min_lat, max_lat])

        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3)
        gl.top_labels = False
        gl.right_labels = False

        # Vẽ Real Track
        ax.plot(real_lon_s, real_lat_s, color='#1f3b5c', linewidth=2.5, label='Observed Track')

        # Vẽ ML Models
        for model, color in zip(plot_models, colors):
            model_df = storm_df[(storm_df["Strategy"] == model) & (storm_df["Horizon"] == horizon)]
            if model_df.empty:
                continue

            lon_s, lat_s = smooth(model_df["PRED_TARGET_LON"], model_df["PRED_TARGET_LAT"])
            
            linestyle = '--' if "CLIPER" in model else '-'
            linewidth = 1.5 if "CLIPER" in model else 2.0
            
            # Format tên model cho gọn
            clean_name = model.split(". ", 1)[-1]
            ax.plot(lon_s, lat_s, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.8, label=clean_name)

        ax.set_title(f"Horizon {horizon}")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), frameon=False)
    plt.suptitle(f"Storm {storm_name} ({sid}): Trajectory Forecast vs Observed", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"../experiments/results/plots/trajectory_{storm_name}_{sid}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ========================================================
    # VẼ BIỂU ĐỒ CƯỜNG ĐỘ GIÓ (WIND INTENSITY)
    # ========================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, horizon in zip(axes, horizons):
        # Real wind
        ax.plot(real_df["time"], real_df["TRUE_TARGET_WIND"], color='black', linewidth=2.5, label='Observed Wind')
        
        for model, color in zip(plot_models, colors):
            model_df = storm_df[(storm_df["Strategy"] == model) & (storm_df["Horizon"] == horizon)]
            if model_df.empty:
                continue
                
            clean_name = model.split(". ", 1)[-1]
            ax.plot(model_df["time"], model_df["PRED_TARGET_WIND"], 
                    color=color, linewidth=2, linestyle='--', alpha=0.8, label=clean_name)
            
        ax.set_title(f"Horizon {horizon}")
        ax.set_ylabel("Wind Intensity (knots)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.5)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15), frameon=False)
    plt.suptitle(f"Storm {storm_name} ({sid}): Wind Intensity Forecast vs Observed", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(f"../experiments/results/plots/wind_intensity_{storm_name}_{sid}.png", dpi=300, bbox_inches='tight')
    plt.close()

print("\nVisualization complete. All plots saved to experiments/results/plots/")