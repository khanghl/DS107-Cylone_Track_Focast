"""
Visualize each of 15 ERA5 channels as separate high-quality images
Mỗi kênh là một hình ảnh riêng biệt
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.gridspec as gridspec

# ════════════════════════════════════════════════════════════════════
# 1. CẤU HÌNH
# ════════════════════════════════════════════════════════════════════

GRIB_PATH = '../data/raw/pressure/2024.grib'
IBTRACS_PATH = '../data/processed/dynamic_feature.csv'

PATCH_SIZE = 25
TARGET_RES = 1.0

VARIABLES = ['z', 'r', 'u', 'v', 'w']
PRESSURE_LEVELS = [200, 500, 850]

VAR_NAMES = {
    'z': 'Geopotential',
    'r': 'Relative Humidity',
    'u': 'U-wind',
    'v': 'V-wind',
    'w': 'Vertical Velocity'
}

VAR_UNITS = {
    'z': 'm²/s²',
    'r': '%',
    'u': 'm/s',
    'v': 'm/s',
    'w': 'Pa/s'
}

VAR_CMAPS = {
    'z': 'RdBu_r',
    'r': 'viridis',
    'u': 'coolwarm',
    'v': 'coolwarm',
    'w': 'PuOr'
}

LEVEL_NAMES = {
    200: '200 hPa',
    500: '500 hPa',
    850: '850 hPa'
}

OUTPUT_DIR = Path('../era5_individual_channels')

# ════════════════════════════════════════════════════════════════════
# 2. CÁC HÀM TIỆN ÍCH (GIỮ NGUYÊN TỪ CODE TRƯỚC)
# ════════════════════════════════════════════════════════════════════

def parse_time_string(time_str: str):
    if isinstance(time_str, pd.Timestamp):
        return time_str
    if isinstance(time_str, str):
        formats = [
            '%Y%m%d_%H%M', '%Y%m%d_%H%M%S',
            '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M', '%Y%m%d%H%M%S', '%Y%m%d%H%M'
        ]
        for fmt in formats:
            try:
                return pd.Timestamp(pd.to_datetime(time_str, format=fmt))
            except (ValueError, TypeError):
                continue
        try:
            return pd.Timestamp(time_str)
        except Exception:
            if '_' in time_str:
                date_part, time_part = time_str.split('_')
                year, month, day = date_part[:4], date_part[4:6], date_part[6:8]
                hour, minute = time_part[:2], time_part[2:4]
                second = time_part[4:6] if len(time_part) >= 6 else '00'
                return pd.Timestamp(f"{year}-{month}-{day} {hour}:{minute}:{second}")
            raise ValueError(f"Không thể parse thời gian: {time_str}")
    return pd.Timestamp(time_str)

def get_storm_position(ibtracs_path: str, storm_id: str = None,
                       year: int = 2024, month: int = 9, day: int = 4, hour: int = 0):
    df = pd.read_csv(ibtracs_path)
    mask = (df['YEAR'] == year) & (df['MONTH'] == month) & \
           (df['DAY'] == day) & (df['HOUR'] == hour)
    if storm_id:
        mask = mask & (df['SID'] == storm_id)
    if len(df[mask]) == 0:
        print(f"⚠️ Không tìm thấy bão tại {year}/{month}/{day} {hour}:00")
        return None
    row = df[mask].iloc[0]
    return {
        'sid': row['SID'],
        'lat': float(row['LAT']),
        'lon': float(row['LON']),
        'time': f"{year:04d}{month:02d}{day:02d}_{hour:02d}00"
    }

def crop_era5_patch_raw(grib_path: str, center_lat: float, center_lon: float,
                        var_name: str, level: int, target_time,
                        patch_size_deg: float = 25.0,
                        target_res: float = 1.0) -> np.ndarray:
    try:
        ds = xr.open_dataset(grib_path, engine='cfgrib')
    except Exception:
        try:
            datasets = xr.open_datasets(grib_path, engine='cfgrib')
            ds = xr.merge(datasets)
        except Exception as e:
            raise ValueError(f"Không thể đọc file GRIB: {e}")

    time_dim = 'time' if 'time' in ds.dims else 'valid_time'
    level_dim = 'isobaricInhPa' if 'isobaricInhPa' in ds.dims else 'level'
    lat_dim = 'latitude' if 'latitude' in ds.dims else 'lat'
    lon_dim = 'longitude' if 'longitude' in ds.dims else 'lon'

    target_timestamp = parse_time_string(target_time)

    try:
        ds_t = ds.sel(**{time_dim: target_timestamp}, method='nearest')
    except Exception:
        try:
            ds_t = ds.sel(**{time_dim: target_timestamp.strftime('%Y-%m-%d %H:%M:%S')}, method='nearest')
        except Exception:
            times = ds[time_dim].values
            if len(times) > 0:
                ds_t = ds.sel(**{time_dim: times[0]}, method='nearest')
            else:
                raise ValueError("Không có dữ liệu thời gian")

    try:
        ds_level = ds_t.sel(**{level_dim: level}, method='nearest')
    except Exception:
        if 'level' in ds_t.dims:
            ds_level = ds_t.sel(level=level, method='nearest')
        else:
            levels = ds_t[level_dim].values
            if len(levels) > 0:
                ds_level = ds_t.sel(**{level_dim: levels[0]}, method='nearest')
            else:
                raise ValueError("Không có dữ liệu mực áp suất")

    if var_name not in ds_level:
        raise ValueError(f"Biến {var_name} không có trong dataset")

    ds_var = ds_level[[var_name]]

    lat_vals = ds_var[lat_dim].values
    lon_vals = ds_var[lon_dim].values

    lat_idx = np.argmin(np.abs(lat_vals - center_lat))
    lon_idx = np.argmin(np.abs(lon_vals - center_lon))

    current_res = abs(float(lat_vals[1] - lat_vals[0]))
    target_pixels = int(round(patch_size_deg / target_res))
    half_pixels = target_pixels // 2

    lat_start = max(0, lat_idx - half_pixels)
    lat_end = min(len(lat_vals), lat_idx + half_pixels + 1)
    lon_start = max(0, lon_idx - half_pixels)
    lon_end = min(len(lon_vals), lon_idx + half_pixels + 1)

    cropped = ds_var.isel(
        **{lat_dim: slice(lat_start, lat_end),
           lon_dim: slice(lon_start, lon_end)}
    )

    data = cropped[var_name].values

    if data.ndim == 3:
        data = data.squeeze()
    elif data.ndim == 4:
        data = data.squeeze()
        if data.ndim > 2:
            data = data[0]

    current_h, current_w = data.shape
    need_h = max(0, target_pixels - current_h)
    need_w = max(0, target_pixels - current_w)

    if need_h > 0 or need_w > 0:
        pad_h_top = need_h // 2
        pad_h_bottom = need_h - pad_h_top
        pad_w_left = need_w // 2
        pad_w_right = need_w - pad_w_left
        data = np.pad(data, pad_width=((pad_h_top, pad_h_bottom), 
                                       (pad_w_left, pad_w_right)), mode='edge')

    if data.shape[0] > target_pixels:
        data = data[:target_pixels, :]
    if data.shape[1] > target_pixels:
        data = data[:, :target_pixels]

    if abs(current_res - target_res) > 0.01 and data.shape[0] > 0:
        from scipy.ndimage import zoom
        factor_h = target_pixels / data.shape[0]
        factor_w = target_pixels / data.shape[1]
        data = zoom(data, (factor_h, factor_w), order=1)

    if data.shape[0] != target_pixels or data.shape[1] != target_pixels:
        if data.shape[0] > target_pixels:
            data = data[:target_pixels, :]
        elif data.shape[0] < target_pixels:
            data = np.pad(data, ((0, target_pixels - data.shape[0]), (0, 0)), mode='edge')
        if data.shape[1] > target_pixels:
            data = data[:, :target_pixels]
        elif data.shape[1] < target_pixels:
            data = np.pad(data, ((0, 0), (0, target_pixels - data.shape[1])), mode='edge')

    return data.astype(np.float32)

# ════════════════════════════════════════════════════════════════════
# 3. HÀM HIỂN THỊ 1 KÊNH RIÊNG BIỆT - CHI TIẾT
# ════════════════════════════════════════════════════════════════════

def visualize_single_channel_detailed(
    data: np.ndarray,
    var_name: str,
    level: int,
    storm_info: dict,
    channel_idx: int,
    save_path: Path = None,
    figsize: tuple = (14, 10)
):
    """
    Hiển thị 1 kênh với đầy đủ thông tin chi tiết
    
    Bao gồm:
    - Heatmap chính
    - Histogram
    - Statistics
    - 3D Surface plot
    - Contour plot
    """
    
    H, W = data.shape
    
    # Tạo figure với subplots
    fig = plt.figure(figsize=figsize)
    
    # Grid layout
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[2, 1])
    
    # --- 1. HEATMAP CHÍNH ---
    ax1 = fig.add_subplot(gs[0, :2])
    cmap = VAR_CMAPS.get(var_name, 'RdBu_r')
    
    # Xác định vmin/vmax để hiển thị tốt
    vmin, vmax = np.percentile(data, [2, 98])
    if vmin == vmax:
        vmin, vmax = data.min(), data.max()
    
    im = ax1.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
    
    # Thêm grid nhẹ
    ax1.set_xticks(np.arange(0, W, 5))
    ax1.set_yticks(np.arange(0, H, 5))
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Thông tin trên ảnh
    title = f"{VAR_NAMES[var_name]} at {LEVEL_NAMES[level]}"
    subtitle = f"Storm: {storm_info['sid']} | Position: ({storm_info['lat']:.2f}°, {storm_info['lon']:.2f}°) | Time: {storm_info['time']}"
    ax1.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label(f'{VAR_NAMES[var_name]} ({VAR_UNITS[var_name]})', fontsize=10)
    
    # --- 2. 3D SURFACE PLOT ---
    ax2 = fig.add_subplot(gs[0, 2], projection='3d')
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)
    
    # Giảm độ phân giải để hiển thị nhanh hơn
    step = max(1, min(H, W) // 20)
    surf = ax2.plot_surface(X[::step, ::step], Y[::step, ::step], 
                           data[::step, ::step], 
                           cmap=cmap, alpha=0.8, linewidth=0, antialiased=True)
    ax2.set_title('3D Surface', fontsize=10)
    ax2.set_xlabel('X', fontsize=8)
    ax2.set_ylabel('Y', fontsize=8)
    ax2.set_zlabel('Value', fontsize=8)
    ax2.view_init(elev=30, azim=45)
    
    # --- 3. HISTOGRAM + STATISTICS ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(data.flatten(), bins=30, alpha=0.7, color='steelblue', 
             edgecolor='black', linewidth=0.5)
    ax3.axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {data.mean():.2f}')
    ax3.axvline(data.mean() + data.std(), color='orange', linestyle=':', 
                linewidth=1.5, label=f'±1σ')
    ax3.axvline(data.mean() - data.std(), color='orange', linestyle=':', linewidth=1.5)
    ax3.set_title('Value Distribution', fontsize=10)
    ax3.set_xlabel('Value', fontsize=8)
    ax3.set_ylabel('Frequency', fontsize=8)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # --- 4. CONTOUR PLOT ---
    ax4 = fig.add_subplot(gs[1, 1])
    contour = ax4.contourf(data, levels=20, cmap=cmap)
    ax4.set_title('Contour Plot', fontsize=10)
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    # --- 5. STATISTICS TABLE ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    stats_text = [
        f"📊 CHANNEL STATISTICS",
        f"",
        f"  Variable: {VAR_NAMES[var_name]}",
        f"  Level: {LEVEL_NAMES[level]}",
        f"  Unit: {VAR_UNITS[var_name]}",
        f"",
        f"  Shape: {data.shape}",
        f"  Min: {data.min():.4f}",
        f"  Max: {data.max():.4f}",
        f"  Mean: {data.mean():.4f}",
        f"  Std: {data.std():.4f}",
        f"  Median: {np.median(data):.4f}",
        f"  Skew: {pd.Series(data.flatten()).skew():.4f}",
        f"  Kurtosis: {pd.Series(data.flatten()).kurtosis():.4f}",
        f"",
        f"  Channel Index: {channel_idx}/14"
    ]
    
    ax5.text(0.05, 0.95, '\n'.join(stats_text), 
             transform=ax5.transAxes,
             fontsize=10, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Đã lưu: {save_path}")
    
    plt.close(fig)
    return fig

# ════════════════════════════════════════════════════════════════════
# 4. HÀM HIỂN THỊ TẤT CẢ 15 KÊNH RIÊNG BIỆT
# ════════════════════════════════════════════════════════════════════

def visualize_all_channels_individual(
    extracted_data: dict,
    storm_info: dict,
    output_dir: Path = OUTPUT_DIR
):
    """
    Hiển thị từng kênh thành ảnh riêng biệt
    """
    channels = extracted_data['channels']
    channel_names = extracted_data['channel_names']
    channel_dict = extracted_data['channel_dict']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("  ĐANG TẠO 15 HÌNH ẢNH RIÊNG BIỆT")
    print("="*60)
    
    # Tạo thư mục con cho từng biến
    for var in VARIABLES:
        var_dir = output_dir / var
        var_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, (key, data) in enumerate(channel_dict.items()):
        var_name = key.split('_')[0]
        level = int(key.split('_')[1])
        
        # Tạo tên file
        filename = f"channel_{idx:02d}_{var_name}_{level}hPa.png"
        save_path = output_dir / var_name / filename
        
        print(f"   🖼️ Đang tạo: {filename}...", end=' ', flush=True)
        
        try:
            visualize_single_channel_detailed(
                data=data,
                var_name=var_name,
                level=level,
                storm_info=storm_info,
                channel_idx=idx,
                save_path=save_path,
                figsize=(14, 10)
            )
            print("✅")
        except Exception as e:
            print(f"❌ {e}")
    
    print("\n" + "="*60)
    print(f"  ✅ HOÀN TẤT! Đã lưu 15 hình ảnh tại: {output_dir}")
    print("="*60)
    
    # In danh sách file
    print("\n📁 Danh sách file đã tạo:")
    for var in VARIABLES:
        var_dir = output_dir / var
        if var_dir.exists():
            files = list(var_dir.glob('*.png'))
            print(f"   📂 {var}/: {len(files)} files")

# ════════════════════════════════════════════════════════════════════
# 5. HÀM HIỂN THỊ 1 KÊNH ĐƠN GIẢN (CHỈ HEATMAP)
# ════════════════════════════════════════════════════════════════════

def visualize_single_channel_simple(
    data: np.ndarray,
    var_name: str,
    level: int,
    storm_info: dict,
    channel_idx: int,
    save_path: Path = None,
    figsize: tuple = (8, 8)
):
    """
    Hiển thị 1 kênh đơn giản chỉ với heatmap và thông tin cơ bản
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    cmap = VAR_CMAPS.get(var_name, 'RdBu_r')
    vmin, vmax = np.percentile(data, [2, 98])
    if vmin == vmax:
        vmin, vmax = data.min(), data.max()
    
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
    
    # Thông tin
    title = f"{VAR_NAMES[var_name]} at {LEVEL_NAMES[level]}"
    subtitle = f"Storm: {storm_info['sid']} | ({storm_info['lat']:.2f}°, {storm_info['lon']:.2f}°) | {storm_info['time']}"
    ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Thêm grid nhẹ
    ax.grid(True, alpha=0.2, linestyle='--')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'{VAR_NAMES[var_name]} ({VAR_UNITS[var_name]})', fontsize=10)
    
    # Thêm statistics
    stats_text = f"Min: {data.min():.2f} | Max: {data.max():.2f} | Mean: {data.mean():.2f} | Std: {data.std():.2f}"
    ax.text(0.5, -0.08, stats_text, transform=ax.transAxes,
            fontsize=9, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Đã lưu: {save_path}")
    
    plt.close(fig)
    return fig


def extract_15_channels_raw(grib_path: str, center_lat: float, center_lon: float,
                            target_time, variables: list[str] = VARIABLES,
                            levels: list[int] = PRESSURE_LEVELS,
                            patch_size_deg: float = 25.0,
                            target_res: float = 1.0) -> dict:
    """
    Trích xuất 15 kênh với giá trị THỰC từ file GRIB
    """
    channels = {}
    channel_list = []
    channel_names = []

    for var in variables:
        for level in levels:
            key = f"{var}_{level}"
            try:
                data = crop_era5_patch_raw(
                    grib_path, center_lat, center_lon,
                    var, level, target_time,
                    patch_size_deg, target_res
                )

                channels[key] = data
                channel_list.append(data)
                channel_names.append(f"{VAR_NAMES[var]}\n{LEVEL_NAMES[level]}")

            except Exception as e:
                print(f"⚠️ Lỗi khi trích xuất {var} tại {level}hPa: {e}")
                # Tạo dữ liệu giả nếu lỗi (để minh họa)
                dummy = np.zeros((25, 25), dtype=np.float32)
                channels[key] = dummy
                channel_list.append(dummy)
                channel_names.append(f"{VAR_NAMES[var]}\n{LEVEL_NAMES[level]}\n(MISSING)")

    return {
        'channels': channel_list,
        'channel_names': channel_names,
        'channel_dict': channels,
        'shape': (len(channel_list), 25, 25)
    }


# ════════════════════════════════════════════════════════════════════
# 6. MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    print("═"*80)
    print("  VISUALIZE 15 ERA5 CHANNELS - INDIVIDUAL IMAGES")
    print("═"*80)
    print()
    
    # --- Bước 1: Lấy vị trí bão ---
    print("📌 Bước 1: Lấy vị trí bão từ IBTrACS...")
    
    # Thử các thời điểm khác nhau
    test_times = [
        (2024, 9, 4, 0),
        (2024, 9, 1, 0),
        (2024, 8, 1, 0),
        (2024, 7, 1, 0)
    ]
    
    storm_info = None
    for year, month, day, hour in test_times:
        storm_info = get_storm_position(
            ibtracs_path=IBTRACS_PATH,
            storm_id=None,
            year=year,
            month=month,
            day=day,
            hour=hour
        )
        if storm_info:
            break
    
    if storm_info is None:
        print("❌ Không tìm thấy bão nào. Vui lòng kiểm tra dữ liệu IBTrACS")
        return
    
    print(f"   ✅ Storm: {storm_info['sid']}")
    print(f"   📍 Position: ({storm_info['lat']:.2f}°, {storm_info['lon']:.2f}°)")
    print(f"   🕐 Time: {storm_info['time']}")
    print()
    
    # --- Bước 2: Trích xuất 15 kênh ---
    print("📌 Bước 2: Trích xuất 15 kênh từ file GRIB...")
    
    extracted = extract_15_channels_raw(
        grib_path=GRIB_PATH,
        center_lat=storm_info['lat'],
        center_lon=storm_info['lon'],
        target_time=storm_info['time'],
        variables=VARIABLES,
        levels=PRESSURE_LEVELS,
        patch_size_deg=25.0,
        target_res=1.0
    )
    
    print(f"   ✅ Đã trích xuất {len(extracted['channels'])} kênh")
    print(f"   📐 Shape: {extracted['shape']}")
    print()
    
    # --- Bước 3: Hiển thị từng kênh riêng biệt ---
    print("📌 Bước 3: Tạo 15 hình ảnh riêng biệt...")
    
    # Chọn chế độ:
    # - 'detailed': ảnh chi tiết (có histogram, 3D, contour)
    # - 'simple': ảnh đơn giản (chỉ heatmap)
    
    MODE = 'detailed'  # Hoặc 'simple'
    
    if MODE == 'detailed':
        visualize_all_channels_individual(extracted, storm_info)
    else:
        # Tạo từng ảnh đơn giản
        output_dir = Path('./era5_simple_channels')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, (key, data) in enumerate(extracted['channel_dict'].items()):
            var_name = key.split('_')[0]
            level = int(key.split('_')[1])
            filename = f"channel_{idx:02d}_{var_name}_{level}hPa.png"
            save_path = output_dir / filename
            
            visualize_single_channel_simple(
                data=data,
                var_name=var_name,
                level=level,
                storm_info=storm_info,
                channel_idx=idx,
                save_path=save_path,
                figsize=(8, 8)
            )
        
        print(f"\n✅ Hoàn tất! Đã lưu tại: {output_dir}")
    
    print("\n✅ HOÀN TẤT!")

if __name__ == '__main__':
    main()