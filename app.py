import io, os, pickle, base64, traceback, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
from PIL import Image
from sklearn.neighbors import NearestNeighbors

PRECOMPUTED_DIR = '/shared/ssd_28T/home/mm3572/anomaly_detection/Mousa/web_app/static/precomputed'

with open(os.path.join(PRECOMPUTED_DIR, 'shared_pca.pkl'), 'rb') as f:
    PCA_SHARED = pickle.load(f)
with open(os.path.join(PRECOMPUTED_DIR, 'dataset_stats.pkl'), 'rb') as f:
    _meta         = pickle.load(f)
    DATASET_STATS = _meta['dataset_stats']
    DATASET_SIZES = _meta['dataset_sizes']
    DATASET_NAMES = _meta['dataset_names']
with open(os.path.join(PRECOMPUTED_DIR, 'embeddings.pkl'), 'rb') as f:
    _emb       = pickle.load(f)
    EMB_TSNE   = _emb['emb_tsne']
    EMB_UMAP   = _emb['emb_umap']
    EMB_LABELS = _emb['labels_all']
    EMB_X_PCA  = _emb['X_pca_all']
    TSNE_MODEL = _emb['tsne_model']
    UMAP_MODEL = _emb['umap_model']

print(f"[app] Loaded {len(DATASET_NAMES)} datasets. t-SNE={EMB_TSNE.shape}")

_unique_labels   = sorted(set(EMB_LABELS))
_combined_colors = []
for _cn in ['tab20', 'tab20b', 'tab20c']:
    _c = plt.get_cmap(_cn)
    _combined_colors.extend([_c(i) for i in range(_c.N)])
COLOR_MAP = {lbl: _combined_colors[i % len(_combined_colors)]
             for i, lbl in enumerate(_unique_labels)}
for k in COLOR_MAP:
    if 'AIROGS' in k:
        COLOR_MAP[k] = plt.get_cmap('tab20')(0)

print('[app] Building kNN index ...')
KNN_INDEX = NearestNeighbors(n_neighbors=15, metric='cosine', n_jobs=-1)
KNN_INDEX.fit(EMB_X_PCA)
print('[app] kNN index ready.')

EPS = 1e-6

def gaussian_kl(mu1, var1, mu2, var2):
    v1, v2 = var1 + EPS, var2 + EPS
    k = len(mu1)
    return 0.5*(np.sum(np.log(v2/v1)) - k + np.sum(v1/v2) + np.sum((mu2-mu1)**2/v2))

def symmetric_kl(mu1, var1, mu2, var2):
    return 0.5*(gaussian_kl(mu1,var1,mu2,var2) + gaussian_kl(mu2,var2,mu1,var1))

def mahalanobis_dist(x, mu, var):
    return float(np.sqrt(np.sum((x - mu)**2 / (var + EPS))))

def _estimate_scale():
    names, vals = DATASET_NAMES, []
    for i in range(len(names)):
        mu_i, var_i = DATASET_STATS[names[i]]
        for j in range(i+1, len(names)):
            mu_j, var_j = DATASET_STATS[names[j]]
            vals.append(symmetric_kl(mu_i, var_i, mu_j, var_j))
    return float(np.median(vals)) if vals else 50.0, float(max(vals)) if vals else 100.0

KL_SCALE, KL_MAX = _estimate_scale()
print(f'[app] KL scale (median) = {KL_SCALE:.2f}  |  KL max = {KL_MAX:.2f}')

def get_features(pil_images):
    from retfound_extractor import extract_features
    return extract_features(pil_images)

def project_tsne(feat_pca):
    if TSNE_MODEL is not None:
        try: return np.array(TSNE_MODEL.transform(feat_pca))
        except: pass
    dists, idxs = KNN_INDEX.kneighbors(feat_pca)
    w = 1.0/(dists+1e-8); w /= w.sum(axis=1, keepdims=True)
    return np.einsum('nk,nkd->nd', w, EMB_TSNE[idxs])

def project_umap(feat_pca):
    if EMB_UMAP is None: return None
    if UMAP_MODEL is not None:
        try: return UMAP_MODEL.transform(feat_pca)
        except: pass
    dists, idxs = KNN_INDEX.kneighbors(feat_pca)
    w = 1.0/(dists+1e-8); w /= w.sum(axis=1, keepdims=True)
    return np.einsum('nk,nkd->nd', w, EMB_UMAP[idxs])

def make_embedding_plot(query_tsne, query_umap, results):
    score_map   = {r['dataset']: r for r in results}
    top_dataset = results[0]['dataset']
    has_umap    = (EMB_UMAP is not None) and (query_umap is not None)
    pairs = [('t-SNE', EMB_TSNE, query_tsne)]
    pairs.append(('UMAP', EMB_UMAP, query_umap) if has_umap
                 else ('UMAP (unavailable)', None, None))

    FS_TITLE    = 48
    FS_AX_LABEL = 38
    FS_TICK     = 30
    FS_CENTROID     = 20
    FS_CENTROID_TOP = 24
    FS_SCOREBOARD   = 22
    FS_SUPTITLE     = 40

    fig, axes = plt.subplots(1, 2, figsize=(48, 20))
    fig.patch.set_facecolor('white')

    for ax, (method, emb_all, q_pos) in zip(axes, pairs):
        ax.set_facecolor('white')
        if emb_all is None:
            ax.text(0.5, 0.5, method, ha='center', va='center',
                    color='#888', fontsize=18, transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        # Scatter datasets
        for lbl in _unique_labels:
            mask   = EMB_LABELS == lbl
            is_top = (lbl == top_dataset)
            ax.scatter(emb_all[mask,0], emb_all[mask,1],
                       color=COLOR_MAP[lbl],
                       alpha=0.75 if is_top else 0.20,
                       s=18 if is_top else 6,
                       zorder=3 if is_top else 1,
                       rasterized=True,
                       label=lbl.replace('dataset_','') + (' (closest)' if is_top else ''))

        # Centroid labels — name only for non-closest, name+MD for closest
        ax.scatter(q_pos[:,0], q_pos[:,1],
                   color='red', marker='*', s=900, zorder=10,
                   edgecolors='black', linewidths=1.0,
                   label='Your image(s)')

        # Compute 2D Mahalanobis for every dataset in this embedding
        mahal_2d_all = {}
        for lbl in _unique_labels:
            pts    = emb_all[EMB_LABELS == lbl]
            mu2    = pts.mean(axis=0)
            var2   = pts.var(axis=0) + 1e-8
            qm     = q_pos.mean(axis=0)
            qm_x   = float(np.clip(qm[0], emb_all[:,0].min(), emb_all[:,0].max()))
            qm_y   = float(np.clip(qm[1], emb_all[:,1].min(), emb_all[:,1].max()))
            mahal_2d_all[lbl] = float(np.sqrt(np.sum(
                (np.array([qm_x, qm_y]) - mu2)**2 / var2)))

        # 2D-closest dataset — used for both arrow and scoreboard
        top_dataset_2d = min(mahal_2d_all, key=mahal_2d_all.get)

        # 2D Mahalanobis arrow to closest cluster
        top_mask   = EMB_LABELS == top_dataset_2d
        top_pts    = emb_all[top_mask]
        top_cx     = float(np.clip(top_pts[:,0].mean(), emb_all[:,0].min(), emb_all[:,0].max()))
        top_cy     = float(np.clip(top_pts[:,1].mean(), emb_all[:,1].min(), emb_all[:,1].max()))
        top_var2d  = top_pts.var(axis=0) + 1e-8
        qmean      = q_pos.mean(axis=0)
        qmean_x    = float(np.clip(qmean[0], emb_all[:,0].min(), emb_all[:,0].max()))
        qmean_y    = float(np.clip(qmean[1], emb_all[:,1].min(), emb_all[:,1].max()))
        mahal_2d   = float(np.sqrt(np.sum((np.array([qmean_x,qmean_y]) - np.array([top_cx,top_cy]))**2 / top_var2d)))
        ax.annotate('', xy=(top_cx, top_cy), xytext=(qmean_x, qmean_y),
                    arrowprops=dict(arrowstyle='->', color='crimson',
                                   lw=2.5, linestyle='dashed'), zorder=9)
        mid_x = (qmean_x + top_cx) / 2
        mid_y = (qmean_y + top_cy) / 2
        dx = top_cx - qmean_x
        dy = top_cy - qmean_y
        length = max(np.sqrt(dx**2 + dy**2), 1e-6)
        perp_x = -dy / length
        perp_y =  dx / length
        offset = (np.ptp(emb_all[:,0]) + np.ptp(emb_all[:,1])) * 0.04
        label_x = mid_x + perp_x * offset
        label_y = mid_y + perp_y * offset
        arrow_txt = 'MD(2D)=' + str(round(mahal_2d,2))
        ax.text(top_cx, top_cy, arrow_txt,
                fontsize=FS_CENTROID_TOP, color='crimson', fontweight='bold',
                ha='center', va='bottom', zorder=12,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='crimson', alpha=0.92, linewidth=1.8))

        top5_2d = sorted(mahal_2d_all.items(), key=lambda x: x[1])[:5]
        board = ['  Top-5 by MD (2D embedding)  ']
        for i, (lbl, md2) in enumerate(top5_2d):
            board.append('  ' + str(i+1) + '. '
                         + lbl.replace('dataset_','').ljust(16)
                         + '  MD(2D)=' + str(round(md2, 2)))
        ax.text(0.01, 0.99, '\n'.join(board),
                transform=ax.transAxes, fontsize=FS_SCOREBOARD,
                va='top', ha='left', family='monospace', color='#1a1a2e',
                bbox=dict(boxstyle='round,pad=0.7', facecolor='#f0f9ff',
                          edgecolor='#93c5fd', alpha=0.95, linewidth=1.5))

        # Legend
        handles, labels_leg = ax.get_legend_handles_labels()
        order = ([i for i,l in enumerate(labels_leg) if 'Your image' in l] +
                 [i for i,l in enumerate(labels_leg) if 'closest' in l] +
                 [i for i,l in enumerate(labels_leg)
                  if 'Your image' not in l and 'closest' not in l])
        handles    = [handles[i] for i in order]
        labels_leg = [labels_leg[i] for i in order]
        ax.legend(handles, labels_leg,
                  loc='upper left', bbox_to_anchor=(1.01, 1.0),
                  fontsize=18, markerscale=5, framealpha=0.9,
                  title='Dataset', title_fontsize=20,
                  borderpad=0.8, labelspacing=0.6)

        method_clean = method.split('\n')[0]
        ax.set_title(method_clean, fontsize=FS_TITLE, fontweight='bold',
                     color='#1a1a2e', pad=16)
        ax.set_xlabel(method_clean + ' Dim 1', fontsize=FS_AX_LABEL, color='#374151')
        ax.set_ylabel(method_clean + ' Dim 2', fontsize=FS_AX_LABEL, color='#374151')
        ax.tick_params(colors='#374151', labelsize=FS_TICK)
        for spine in ax.spines.values():
            spine.set_edgecolor('#374151')
            spine.set_linewidth(2.5)
        ax.grid(True, linewidth=1.2, alpha=0.6, color='#9ca3af')

    top_r    = score_map[top_dataset]
    top_name = top_dataset.replace('dataset_','')
    kl_str   = ('  |  KL=' + str(round(top_r['kl_divergence'],2))
                if top_r['kl_divergence'] is not None else '')
    sup = ('Feature Space Embedding  --  Your Image(s) vs '
           + str(len(_unique_labels)) + ' Reference Datasets\n'
           + 'Closest match:  ' + top_name
           + '   (MD=' + str(round(top_r['mahalanobis'],2)) + kl_str + ')   red *')
    fig.suptitle(sup, fontsize=FS_SUPTITLE, color='#1a1a2e', y=1.02, fontweight='bold')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
ALLOWED_EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

@app.route('/')
def index():
    return render_template('index.html', n_datasets=len(DATASET_NAMES))

@app.route('/query', methods=['POST'])
def query():
    files = request.files.getlist('images')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No images uploaded.'}), 400

    pil_images, filenames, bad_files = [], [], []
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_EXT:
            bad_files.append(f.filename); continue
        try:
            pil_images.append(Image.open(io.BytesIO(f.read())).convert('RGB'))
            filenames.append(f.filename)
        except Exception:
            bad_files.append(f.filename)

    if not pil_images:
        return jsonify({'error': 'No valid images could be read.'}), 400

    try:
        feats     = get_features(pil_images)
        feats_pca = PCA_SHARED.transform(feats)
        n_imgs    = feats_pca.shape[0]
        single    = (n_imgs == 1)

        q_mu = feats_pca.mean(axis=0)
        if not single:
            q_var = feats_pca.var(axis=0)
        else:
            all_vars = np.stack([var for _, var in DATASET_STATS.values()])
            q_var    = all_vars.mean(axis=0)

        kl_values, mahal_values = {}, {}
        for name in DATASET_NAMES:
            mu_d, var_d = DATASET_STATS[name]
            kl_values[name]    = float(symmetric_kl(q_mu, q_var, mu_d, var_d))
            mahal_values[name] = float(np.mean([mahalanobis_dist(feats_pca[i], mu_d, var_d)
                                                for i in range(n_imgs)]))

        kl_min,    kl_max    = min(kl_values.values()),    max(kl_values.values())
        mahal_min, mahal_max = min(mahal_values.values()), max(mahal_values.values())

        def rel_sim(val, vmin, vmax):
            return round((1.0 - (val-vmin)/(vmax-vmin+1e-12))*100.0, 2)

        results = []
        for name in DATASET_NAMES:
            results.append({
                'dataset'         : name,
                'display_name'    : name.replace('dataset_',''),
                'kl_divergence'   : round(kl_values[name], 4) if not single else None,
                'kl_similarity'   : rel_sim(kl_values[name], kl_min, kl_max) if not single else None,
                'mahalanobis'     : round(mahal_values[name], 4),
                'mahal_similarity': rel_sim(mahal_values[name], mahal_min, mahal_max),
                'n_samples'       : DATASET_SIZES.get(name, '?'),
            })
        results.sort(key=lambda x: x['mahalanobis'])

        per_image_rows = []
        for i, fname in enumerate(filenames):
            img_mahal = {n: mahalanobis_dist(feats_pca[i], DATASET_STATS[n][0], DATASET_STATS[n][1])
                         for n in DATASET_NAMES}
            closest = min(img_mahal, key=img_mahal.get)
            per_image_rows.append({
                'image_name'      : fname,
                'closest_dataset' : closest.replace('dataset_',''),
                'mahalanobis'     : round(img_mahal[closest], 4),
                'mahal_similarity': rel_sim(img_mahal[closest], mahal_min, mahal_max),
                'kl_divergence'   : round(kl_values[closest], 4) if not single else 'N/A',
                'kl_normalized'   : round(float(np.log1p(kl_values[closest]) / np.log1p(KL_MAX)), 6) if not single else 'N/A',
            })

        csv_buf = io.StringIO()
        writer  = csv.DictWriter(csv_buf, fieldnames=[
            'image_name','closest_dataset','mahalanobis','mahal_similarity_%',
            'kl_divergence','kl_normalized'])
        writer.writeheader()
        for row in per_image_rows:
            writer.writerow({'image_name': row['image_name'],
                             'closest_dataset': row['closest_dataset'],
                             'mahalanobis': row['mahalanobis'],
                             'mahal_similarity_%': row['mahal_similarity'],
                             'kl_divergence': row['kl_divergence'],
                             'kl_normalized': row['kl_normalized']})
        csv_b64 = base64.b64encode(csv_buf.getvalue().encode()).decode()

        q_tsne  = project_tsne(feats_pca)
        q_umap  = project_umap(feats_pca)
        emb_b64 = make_embedding_plot(q_tsne, q_umap, results)

        return jsonify({
            'n_images_processed': n_imgs,
            'single_image'      : single,
            'bad_files'         : bad_files,
            'results'           : results,
            'per_image_rows'    : per_image_rows,
            'embedding_b64'     : emb_b64,
            'csv_b64'           : csv_b64,
            'top_match'         : results[0],
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
