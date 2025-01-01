import numpy as np
import json

def extract_protein_centers(json_file, scale):
    """
    JSONデータからタンパク質名と中心座標を抽出する。
    
    Parameters:
        json_file (str): JSONファイルのパス。
    
    Returns:
        dict: タンパク質名をキーとした中心座標のリスト。
    """
    # JSONファイルを読み込む
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # タンパク質名を取得
    protein_name = data.get("pickable_object_name")
    
    # 中心座標を取得
    points = data.get("points", [])
    centers = [
        (
            point["location"]["z"] / scale,
            point["location"]["y"] / scale,
            point["location"]["x"] / scale
        ) for point in points
    ]
    
    # 結果を辞書形式で返す
    return {protein_name: centers}


def generate_gaussian_heatmap(shape, center, sigma):
    """
    3Dガウス分布のヒートマップを生成する。
    
    Parameters:
        shape (tuple): 3Dトモグラムのサイズ (depth, height, width)
        center (tuple): ガウス分布の中心 (z, y, x)
        sigma (float): ガウス分布の標準偏差
    
    Returns:
        numpy.ndarray: 3Dヒートマップ
    """
    z, y, x = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij'
    )
    dist_squared = (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2
    heatmap = np.exp(-dist_squared / (2 * sigma ** 2))
    return heatmap

def generate_protein_heatmap(shape, centers, sigma):
    """
    1つのたんぱく質に対するヒートマップを生成する。
    
    Parameters:
        shape (tuple): 3Dトモグラムのサイズ
        centers (list): 各中心座標のリスト [(z1, y1, x1), (z2, y2, x2), ...]
        sigma (float): ガウス分布の標準偏差
    
    Returns:
        numpy.ndarray: 3Dヒートマップ
    """
    heatmap = np.zeros(shape, dtype=np.float32)
    for center in centers:
        heatmap += generate_gaussian_heatmap(shape, center, sigma)
    heatmap = np.clip(heatmap, 0, 1)  # 値を0～1に正規化
    return heatmap

def prepare_ground_truth(shape, protein_centers, sigma):
    """
    データセット全体のGround Truthを準備する。
    
    Parameters:
        shape (tuple): 3Dトモグラムのサイズ
        protein_centers (dict): たんぱく質ごとの中心座標 {protein_id: [(z, y, x), ...]}
        sigma (float): ガウス分布の標準偏差
    
    Returns:
        numpy.ndarray: (6, depth, height, width)の3Dヒートマップ
    """
    num_proteins = len(protein_centers)
    gt_volume = np.zeros((num_proteins, *shape), dtype=np.float32)
    
    for i, (protein_id, centers) in enumerate(protein_centers.items()):
        gt_volume[i] = generate_protein_heatmap(shape, centers, sigma)
    
    return gt_volume

def prepare_ground_truth_with_config(shape, protein_centers, r_scale, config):
    """
    設定情報に基づいてGTを生成する。
    
    Parameters:
        shape (tuple): 3Dトモグラムのサイズ
        protein_centers (dict): 各タンパク質の中心座標 {label: [(z, y, x), ...]}
        config (list): タンパク質の設定情報
    
    Returns:
        numpy.ndarray: (6, depth, height, width)の3Dヒートマップ
    """
    gt_volume = np.zeros((6, *shape), dtype=np.float32)

    for protein in config.pickable_objects:
        label = protein.label
        radius = protein.radius / r_scale
        centers = protein_centers[protein.name]

        for center in centers:
            gt_volume[label - 1] += generate_gaussian_heatmap(shape, center, radius)
    
        # 値を0～1にクリップ
        gt_volume[label - 1] = np.clip(gt_volume[label - 1], 0, 1)

    return gt_volume