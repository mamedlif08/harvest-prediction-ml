"""
chcp 65001
>> $env:PYTHONIOENCODING="utf-8"
>> $OutputEncoding = [System.Text.Encoding]::UTF16
>> python main.py | Tee-Object -FilePath run_log.txt
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from shapely import wkb, wkt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# Ниже — те же вспомогательные функции, что и в train_and_evaluate.py
# (скопированы без изменений, чтобы скрипт был самодостаточным)
# ---------------------------------------------------------------------------

def extract_geo_features_from_source(geom_field):
    try:
        polygon = extract_polygon(geom_field)
        if polygon is not None and not polygon.is_empty:
            centroid = polygon.centroid
            return pd.Series({
                'centroid_x': centroid.x,
                'centroid_y': centroid.y,
                'area': polygon.area
            })
    except Exception:
        pass
    return pd.Series({'centroid_x': 0, 'centroid_y': 0, 'area': 0})


def extract_polygon(geom_field):
    if geom_field is None:
        return None
    try:
        if isinstance(geom_field, dict):
            if 'wkb' in geom_field:
                return wkb.loads(geom_field['wkb'])
            elif 'wkt' in geom_field:
                return wkt.loads(geom_field['wkt'])
        elif isinstance(geom_field, bytes):
            return wkb.loads(geom_field)
        elif isinstance(geom_field, str):
            return wkt.loads(geom_field)
    except Exception:
        pass
    return None


def load_extra_feature_file(path, geozone_id_col_hint='geozone_id'):
    if not os.path.exists(path):
        print(f"⚠️  Файл с фичами не найден: {path} — пропускаю этот блок фич.")
        return None

    if path.lower().endswith(('.xlsx', '.xls')):
        extra = pd.read_excel(path)
    else:
        extra = pd.read_csv(path)

    geo_col = next(
        (c for c in extra.columns if c.lower().replace('_', '').startswith('geozoneid')),
        None
    )
    if geo_col is None:
        raise ValueError(f"Не нашёл колонку geozone_id в файле {path}. Колонки: {list(extra.columns)}")

    if geo_col != geozone_id_col_hint:
        extra = extra.rename(columns={geo_col: geozone_id_col_hint})

    return extra


def merge_extra_features(df, geozone_id_col, feature_files, keep_one_is_synthetic=True):
    added_feature_cols = []
    is_synthetic_cols = []

    for path in feature_files:
        extra = load_extra_feature_file(path)
        if extra is None:
            continue

        if 'is_synthetic' in extra.columns:
            is_synthetic_cols.append(extra[['geozone_id', 'is_synthetic']].rename(
                columns={'is_synthetic': f'is_synthetic__{os.path.basename(path)}'}
            ))
            extra = extra.drop(columns=['is_synthetic'])

        new_cols = [c for c in extra.columns if c != 'geozone_id']
        added_feature_cols.extend(new_cols)

        df = df.merge(extra, on='geozone_id', how='left')

        for c in new_cols:
            if df[c].isna().any():
                median_val = df[c].median()
                n_missing = df[c].isna().sum()
                print(f"   ⚠️  {n_missing} строк без значения '{c}' — заполняю медианой ({median_val:.4f})")
                df[c] = df[c].fillna(median_val)

    if keep_one_is_synthetic and is_synthetic_cols:
        merged_flags = is_synthetic_cols[0]
        for extra_flags in is_synthetic_cols[1:]:
            merged_flags = merged_flags.merge(extra_flags, on='geozone_id', how='outer')
        flag_cols = [c for c in merged_flags.columns if c != 'geozone_id']
        merged_flags['is_synthetic'] = merged_flags[flag_cols].any(axis=1)
        df = df.merge(merged_flags[['geozone_id', 'is_synthetic']], on='geozone_id', how='left')
        df['is_synthetic'] = df['is_synthetic'].fillna(False)

    return df, added_feature_cols


def load_and_prepare():
    """Полностью повторяет подготовку данных из train_and_evaluate.py, без обучения."""
    csv_path = "cotton_geozones_az.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Файл {csv_path} не найден!")

    df = pd.read_csv(csv_path)

    geozone_id_col = next((c for c in ['geozone_id', 'geozoneId'] if c in df.columns), 'geozone_id')
    crop_id_col = next((c for c in ['crop_id', 'cropId'] if c in df.columns), 'crop_id')

    geom_col = 'geozone_geometry' if 'geozone_geometry' in df.columns else 'geometry'
    geo_features = df[geom_col].apply(extract_geo_features_from_source)
    df = pd.concat([df, geo_features], axis=1)

    sowing_col = next((c for c in ['sowing_time', 'sowing_date', 'date_sowing'] if c in df.columns), 'sowing_date')
    harvest_col = next((c for c in ['harvest_time', 'actual_harvest_date', 'harvest_date'] if c in df.columns), 'actual_harvest_date')

    df['sowing_date'] = pd.to_datetime(df[sowing_col])
    df['actual_harvest_date'] = pd.to_datetime(df[harvest_col])

    df['sowing_dayofyear'] = df['sowing_date'].dt.dayofyear
    df['sowing_month'] = df['sowing_date'].dt.month

    df['target_duration_days'] = (df['actual_harvest_date'] - df['sowing_date']).dt.days
    df = df.dropna(subset=['target_duration_days'])

    crop_dummies = pd.get_dummies(df[crop_id_col], prefix='crop')
    df = pd.concat([df, crop_dummies], axis=1)
    crop_feature_cols = list(crop_dummies.columns)

    extra_feature_files = [
        "climate_features_SYNTHETIC.csv",
        "vegetation_indices_SYNTHETIC.csv",
    ]
    if geozone_id_col != 'geozone_id':
        df = df.rename(columns={geozone_id_col: 'geozone_id'})
        geozone_id_col = 'geozone_id'
    df, extra_feature_cols = merge_extra_features(df, geozone_id_col, extra_feature_files)

    return df, crop_id_col, geozone_id_col, crop_feature_cols, extra_feature_cols


# ---------------------------------------------------------------------------
# Диагностика
# ---------------------------------------------------------------------------

def run_diagnostics():
    df, crop_id_col, geozone_id_col, crop_feature_cols, extra_feature_cols = load_and_prepare()

    if not extra_feature_cols:
        print("Не нашёл ни одной climate/vegetation фичи после мёрджа — проверь пути к SYNTHETIC-файлам.")
        return

    # --- 1) Прямая корреляция с таргетом (весь датасет) -----------------------
    print("=" * 70)
    print("1) Корреляция climate/vegetation фич с target_duration_days (весь датасет)")
    print("=" * 70)
    corr_all = df[extra_feature_cols + ['target_duration_days']].corr()['target_duration_days']
    corr_all = corr_all.drop('target_duration_days').sort_values(key=lambda s: s.abs(), ascending=False)
    print(corr_all.to_string())
    print("\nЕсли все значения около 0 (скажем, |corr| < 0.05) — сигнала в этих фичах")
    print("почти нет ещё ДО всякого дерева, и низкая importance в XGBoost — это правда, а не баг.\n")

    # --- 2) Корреляция внутри каждого crop_id (остаточный сигнал) -------------
    print("=" * 70)
    print("2) Корреляция climate/vegetation фич с таргетом ВНУТРИ каждого crop_id")
    print("=" * 70)
    for crop_val, sub in df.groupby(crop_id_col):
        if len(sub) < 20:
            continue
        corr_sub = sub[extra_feature_cols].corrwith(sub['target_duration_days'])
        corr_sub = corr_sub.dropna().sort_values(key=lambda s: s.abs(), ascending=False)
        print(f"\ncrop_id = {crop_val} (n={len(sub)}), топ-3 по |corr|:")
        print(corr_sub.head(3).to_string())

    # --- 3) R² модели с crop-дамми vs без них ---------------------------------
    print("\n" + "=" * 70)
    print("3) Сравнение R²: с crop-дамми vs без crop-дамми (только geo/date + climate/veg)")
    print("=" * 70)

    base_features = ['centroid_x', 'centroid_y', 'area', 'sowing_dayofyear', 'sowing_month']

    def fit_eval(feature_cols, label, save_predictions=False):
        X = df[feature_cols]
        y = df['target_duration_days']
        idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=42)
        X_train, X_test = X.loc[idx_train], X.loc[idx_test]
        y_train, y_test = y.loc[idx_train], y.loc[idx_test]

        model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        print(f"{label}: MAE={mae:.2f} дней, R²={r2:.3f}")

        if save_predictions:
            result_df = pd.DataFrame({
                'geozone_id': df.loc[idx_test, geozone_id_col].values,
                'sowing_date': df.loc[idx_test, 'sowing_date'].dt.strftime('%Y-%m-%d').values,
                'harvest_date': df.loc[idx_test, 'actual_harvest_date'].dt.strftime('%Y-%m-%d').values,
                'predicted_harvest_date': [
                    (sowing + pd.Timedelta(days=int(round(d)))).strftime('%Y-%m-%d')
                    for sowing, d in zip(df.loc[idx_test, 'sowing_date'], pred)
                ]
            }).reset_index(drop=True)
            out_path = "predictions_test.csv"
            result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"   → предикты сохранены в {out_path} ({len(result_df)} строк)")

        return model, feature_cols

    # (a) полный набор — с crop. Это та же модель, что в основном train_and_evaluate.py,
    # поэтому именно для неё сохраняем предикты.
    fit_eval(base_features + crop_feature_cols + extra_feature_cols,
              "С crop-дамми + climate/veg", save_predictions=True)

    # (b) без crop-дамми — оставляем только geo/date + climate/veg
    fit_eval(base_features + extra_feature_cols, "БЕЗ crop-дамми (только geo/date + climate/veg)")

    # (c) только crop-дамми — сколько объясняет один только сорт
    fit_eval(crop_feature_cols, "Только crop-дамми (без geo/date/climate/veg)")


if __name__ == "__main__":
    run_diagnostics()