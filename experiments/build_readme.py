"""Generate theoretical model diagrams and refresh README from local results.

Save as experiments/build_readme.py; run from the repository root:
    python3 -m experiments.build_readme
No training or experimental CSV modification is performed.
"""
from pathlib import Path
import csv
import math
import re
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / 'results' / 'figures'
TABLES = ROOT / 'results' / 'tables'
REPORTS = ROOT / 'results' / 'classification_reports'
HISTORIES = ROOT / 'results' / 'histories'

# Existing plot filenames, their order, and the interpretation supported by the
# recorded experimental summaries. No curve data are invented.
PLOTS = {
    'baseline': {
        'training_loss': 'Training loss across epochs. Compare the rate of objective reduction; a low training loss alone does not establish generalization.',
        'validation_loss': 'Validation cross entropy. SGD has the lowest final baseline loss; Adam and RMSprop finish close to ln(10). The logarithmic axis accommodates large values.',
        'training_accuracy': 'Training accuracy over the full budget; compare with validation accuracy to distinguish fitting from generalization.',
        'validation_accuracy': 'Four optimizers end above 90%, while Adam and RMSprop end at 10%. Adam initially reaches the target and later collapses.',
        'gradient_norm': 'Mean global gradient L2 norm by epoch (log scale). Large or vanishing norms are diagnostic indicators, not proof of a causal mechanism.',
        'training_stability': 'Within-epoch batch-loss standard deviation (log scale); this measures loss variability, not gradient variance directly.',
        'best_accuracy': 'Adagrad has the highest baseline peak accuracy (92.88%); Momentum is only 0.06 pp behind.',
        'final_accuracy': 'Adagrad leads final accuracy (92.84%); Adam and RMSprop finish at chance-level accuracy.',
        'training_time': 'SGD has the lowest measured baseline training time (46.97 s) with batch size and epoch count matched.',
        'gpu_memory': 'SGD has the lowest allocated baseline memory; Adam has the largest, consistent with its additional optimizer state.'},
    'learning_rate': {
        'best_accuracy_vs_lr': 'Peak accuracy is LR-sensitive: SGD peaks at 0.1, whereas Adam and RMSprop peak at 0.001 among tested values.',
        'final_accuracy_vs_lr': 'Final accuracy exposes the finite collapse of Adam and RMSprop at 0.01 and 0.1. Missing points correspond to numerical failures.',
        'validation_loss_vs_lr': 'The very high RMSprop loss at 0.1 can dominate the linear scale. A low final loss and a high peak accuracy need not occur at the same LR.',
        'convergence_vs_lr': 'Epoch of first validation accuracy at or above 85%. Missing points are unachieved targets or failed runs, not zero epochs.'},
    'momentum': {
        'best_accuracy_vs_momentum': 'Momentum SGD peaks at 0.95 and Nesterov at 0.9; the peak difference between Momentum 0.9 and 0.95 is only 0.01 pp.',
        'final_accuracy_vs_momentum': 'Both methods achieve their best final accuracy at 0.9 and deteriorate at 0.99.',
        'loss_vs_momentum': 'Momentum 0.5 gives the lowest final cross entropy in both families, despite not maximizing accuracy.',
        'time_vs_momentum': 'Runtime changes little with the coefficient because the number of updates and buffer shape remain fixed.'},
    'batch_size': {
        'best_accuracy': 'Adagrad at BS=32 leads the sweep with 93.26% peak accuracy; the best batch differs by optimizer.',
        'final_accuracy': 'Adagrad at BS=32 also leads final accuracy. RMSprop at BS=32 has a high peak but a lower final value.',
        'validation_loss': 'Loss compares probabilistic fit rather than only the top-1 decision; larger batches may improve loss without maximizing accuracy.',
        'training_time': 'All six configurations train faster with larger batches under the same epoch budget.',
        'gpu_memory': 'Allocated memory increases with batch size because more activations are retained per update.',
        'throughput': 'All six process more images per second as batch size increases; SGD at BS=256 reaches approximately 22,089 images/s.',
        'updates_to_target': 'Larger batches require fewer updates per epoch; this is an epoch-end upper bound on updates to the target, not an exact crossing measurement.',
        'convergence_time': 'First-target time is generally lower with larger batches, except when the target requires more epochs (e.g., SGD at BS=256).'},
    'adaptive': {
        'rmsprop_alpha': 'Alpha 0.99 maximizes peak and final accuracy; 0.999 is slower to reach the target but has lower final loss.',
        'adam_beta1': 'Beta1=0.8 produces the highest peak and final accuracy at fixed beta2=0.999.',
        'adam_beta2': 'Beta2=0.9999 gives the highest peak and final accuracy at fixed beta1=0.9. The two best individual beta values were not tested jointly.'},
    'epochs': {
        'best_accuracy': 'Best-so-far accuracy is non-decreasing by construction. Momentum and Nesterov show no further peak gains after the 10-epoch budget.',
        'final_accuracy': 'End-of-budget accuracy can fall even when best-so-far accuracy is unchanged; SGD drops to 88.74% at epoch 10.',
        'training_loss': 'The recorded training loss decreases across sampled budgets for all six selected runs.',
        'validation_loss': 'All six selected runs have higher final validation loss at 20 than at 5 epochs; Adagrad improves accuracy but worsens cross entropy.',
        'generalization_gap': 'Training-minus-validation accuracy gap widens between 5 and 20 epochs for all six runs, consistent with overfitting.',
        'training_time': 'Training cost grows approximately with the epoch budget. Adagrad spends another 44.85 s for only 0.12 pp peak gain from 10 to 20 epochs.'},
    'stability': {
        'best_accuracy_stability': 'Mean peak validation accuracy ± sample SD. Adagrad leads the mean; Momentum has the smallest peak SD.',
        'final_accuracy_stability': 'Mean final accuracy ± sample SD. Adagrad leads both the mean and the repeatability of final accuracy.',
        'training_time_stability': 'Mean training time ± sample SD; small-batch Adagrad and RMSprop take longer. Different batch sizes confound isolated optimizer cost.'},
    'final': {
        'test_accuracy': 'Mean independent test accuracy ± sample SD across three seeds. Adagrad leads the observed mean; RMSprop has the smallest SD. The displayed accuracy axis is truncated for readability.',
        'validation_vs_test': 'Mean best validation accuracy is higher than mean test accuracy for all selected configurations. Validation-based selection can contribute to this gap.',
        'classification_metrics': 'Macro precision, recall and F1. Adagrad leads the three metrics; macro recall equals overall accuracy on this balanced test set. The displayed axis is truncated.',
        'accuracy_time_tradeoff': 'Accuracy versus mean training time; marker area represents allocated memory. Adam offers near-Adagrad accuracy with much less training time, at a larger measured memory allocation.',
        'test_loss': 'Momentum has the lowest mean cross entropy; RMSprop has the highest. Loss and accuracy rank differently.',
        'training_time': 'Momentum is the fastest selected configuration on average; Adagrad and RMSprop use more updates due to BS=32.',
        'gpu_memory': 'Adagrad and RMSprop have the lowest measured memory among selected configurations because they use the smallest batches. This is not an isolated optimizer-state comparison.',
        'test_stability': 'RMSprop has the lowest test accuracy SD, followed by Momentum. Three seeds yield descriptive repeatability rather than a statistical significance claim.',
        'convergence_time': 'Momentum has the shortest mean time to the first recorded 85% validation accuracy among the selected configurations.'}
}
ARCH_NAMES = {'cnn_architecture', 'leaky_relu', 'gelu', 'activation_comparison', 'activation_derivatives'}
BASELINE_METRICS = {'training_loss': 'train_loss', 'validation_loss': 'val_loss', 'training_accuracy': 'train_accuracy', 'validation_accuracy': 'val_accuracy', 'gradient_norm': 'global_gradient_norm', 'training_stability': 'batch_loss_std'}


def generate_architecture_figures(root=ROOT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyBboxPatch

    out = root / 'results' / 'figures' / 'architecture'
    out.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-5, 5, 1001)
    Phi = 0.5 * (1 + np.array([math.erf(t / math.sqrt(2)) for t in x]))
    phi = np.exp(-x**2 / 2) / math.sqrt(2 * math.pi)
    relu = np.maximum(x, 0)
    leaky = np.where(x >= 0, x, 0.01 * x)
    gelu = x * Phi
    relu_derivative = np.where(x > 0, 1.0, np.where(x < 0, 0.0, np.nan))
    leaky_derivative = np.where(x > 0, 1.0, np.where(x < 0, 0.01, np.nan))
    gelu_derivative = Phi + x * phi

    def line_plot(filename, title, ylabel, curves):
        fig, ax = plt.subplots(figsize=(9, 5.2))
        for label, values in curves:
            ax.plot(x, values, linewidth=2, label=label)
        ax.axhline(0, color='0.6', linewidth=0.8)
        ax.axvline(0, color='0.6', linewidth=0.8)
        ax.set(xlabel='Input x', ylabel=ylabel, title=title)
        ax.grid(alpha=0.22)
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(out / (filename + '.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    line_plot('leaky_relu', 'LeakyReLU (negative slope = 0.01)', 'f(x)', [('LeakyReLU', leaky)])
    line_plot('gelu', 'Gaussian Error Linear Unit (exact GELU)', 'f(x)', [('GELU', gelu)])
    line_plot('activation_comparison', 'Activation function comparison', 'f(x)', [('ReLU (reference)', relu), ('LeakyReLU', leaky), ('GELU', gelu)])
    line_plot('activation_derivatives', 'Activation derivatives', "f'(x)", [('ReLU (reference)', relu_derivative), ('LeakyReLU', leaky_derivative), ('GELU', gelu_derivative)])

    fig, ax = plt.subplots(figsize=(12, 8.1))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.2)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.text(6, 7.85, 'Leaky-GELU CNN', ha='center', va='center', fontsize=22, weight='bold', color='#142b42')
    ax.text(6, 7.42, 'Fashion-MNIST | 475,434 trainable parameters | 10 output logits', ha='center', fontsize=11, color='#44566a')
    nodes = [
        (2, 6.45, 'Input', '1 × 28 × 28', '#e5edf5'),
        (6, 6.45, 'Conv block 1', '1→32→32 | 3×3\nLeakyReLU ×2', '#d9eee9'),
        (10, 6.45, 'MaxPool 1', '2×2, stride 2\n32 × 14 × 14', '#e5edf5'),
        (10, 4.55, 'Conv block 2', '32→64→64 | 3×3\nLeakyReLU ×2', '#d9eee9'),
        (6, 4.55, 'MaxPool 2', '2×2, stride 2\n64 × 7 × 7', '#e5edf5'),
        (2, 4.55, 'Flatten', '3,136 features', '#e5edf5'),
        (2, 2.65, 'Linear 1', '3,136 → 128\nGELU', '#f8e8cc'),
        (6, 2.65, 'Linear 2', '128 → 64\nGELU', '#f8e8cc'),
        (10, 2.65, 'Linear 3', '64 → 10\nRaw logits', '#f8e8cc'),
    ]
    for cx, cy, title, subtitle, color in nodes:
        patch = FancyBboxPatch((cx-1.67, cy-0.62), 3.34, 1.24, boxstyle='round,pad=0.03,rounding_size=0.13', facecolor=color, edgecolor='#52687e', linewidth=1.1)
        ax.add_patch(patch)
        ax.text(cx, cy+0.22, title, ha='center', va='center', fontsize=12.5, weight='bold', color='#142b42')
        ax.text(cx, cy-0.19, subtitle, ha='center', va='center', fontsize=10.5, color='#344b5e')
    for a, b in zip(nodes, nodes[1:]):
        x1, y1 = a[:2]
        x2, y2 = b[:2]
        if y1 == y2:
            d = 1 if x2 > x1 else -1
            p1, p2 = (x1+d*1.73, y1), (x2-d*1.73, y2)
        else:
            p1, p2 = (x1, y1-0.69), (x2, y2+0.69)
        ax.annotate('', xy=p2, xytext=p1, arrowprops={'arrowstyle':'->', 'lw':1.6, 'color':'#52687e'})
    ax.text(6, 0.95, 'Training: CrossEntropyLoss on logits  |  Inference: argmax or optional softmax', ha='center', fontsize=12, color='#142b42', weight='bold')
    ax.text(6, 0.48, 'Convolutions: stride 1, padding 1  •  No batch normalization or dropout', ha='center', fontsize=10.5, color='#44566a')
    fig.savefig(out / 'cnn_architecture.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def relative(path):
    return quote(path.relative_to(ROOT).as_posix(), safe='/')


def cell(value):
    if value is None or str(value).strip().lower() in ('', 'nan', 'none', 'null'):
        return '—'
    value = str(value)
    try:
        number = float(value)
        if math.isfinite(number) and ('.' in value or 'e' in value.lower()):
            # Compact display only; original precision is in the CSV link.
            value = format(number, '.7g')
    except ValueError:
        pass
    return value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('|', '&#124;').replace('\n', '<br>')


def md_table(header, rows):
    return '\n'.join(['| ' + ' | '.join(map(cell, header)) + ' |', '| ' + ' | '.join('---' for _ in header) + ' |'] + ['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows])


def records(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def csv_block(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))
    if not rows:
        return f'[{path.name}]({relative(path)}) — empty CSV.'
    header = rows[0]
    header = [name or 'Class / metric' for name in header]
    return '\n'.join([f'<details><summary>{path.name} — {len(rows)-1} rows</summary>', '', f'[Open original CSV]({relative(path)})', '', md_table(header, rows[1:]), '', '</details>'])


def replace_block(doc, name, content):
    start, end = f'<!-- BEGIN {name} -->', f'<!-- END {name} -->'
    if doc.count(start) != 1 or doc.count(end) != 1:
        raise ValueError(f'Missing or duplicated README markers: {name}')
    left, rest = doc.split(start, 1)
    _, right = rest.split(end, 1)
    return left + start + '\n' + content.strip() + '\n' + end + right


def title(stem):
    return stem.replace('_', ' ').replace('rmsprop', 'RMSprop').replace('adam', 'Adam').title()


def baseline_detail(stem):
    metric = BASELINE_METRICS.get(stem)
    if not metric:
        return ''
    rows = []
    for opt in ('sgd','momentum','nesterov','adagrad','rmsprop','adam'):
        path = HISTORIES / f'baseline_{opt}_lr0.01_bs128_ep20_seed42' / 'history.csv'
        if not path.exists():
            continue
        try:
            values = [(int(float(r['epoch'])), float(r[metric])) for r in records(path) if r.get(metric)]
            values = [(e,v) for e,v in values if math.isfinite(v)]
            if not values:
                continue
            peak = max(values, key=lambda item: item[1])
            low = min(values, key=lambda item: item[1])
            rows.append([opt.upper(), f'{values[0][1]:.5g}', f'{values[-1][1]:.5g}', f'{low[1]:.5g} (E{low[0]})', f'{peak[1]:.5g} (E{peak[0]})'])
        except (KeyError, ValueError):
            continue
    return ('\n\n<details><summary>Measured curve statistics</summary>\n\n' + md_table(['Optimizer','First epoch','Last epoch','Minimum (epoch)','Maximum (epoch)'], rows) + '\n\n</details>') if rows else ''


def figure_block(path, caption, detail=''):
    heading = title(path.stem)
    return f'#### {heading}\n\n![{heading}]({relative(path)})\n\n*{caption}*{detail}'


def confusion_analysis():
    if not REPORTS.exists():
        return 'No classification-report directory was found.'
    parts = []
    for opt in ('sgd','momentum','nesterov','adagrad','rmsprop','adam'):
        matrices = sorted(REPORTS.glob(f'{opt}_seed*_confusion.csv'))
        if not matrices:
            continue
        labels = None
        counts = None
        for path in matrices:
            with path.open(newline='', encoding='utf-8-sig') as f:
                raw = list(csv.reader(f))
            if not raw:
                raise ValueError(f'Empty confusion matrix: {path}')
            current_labels = raw[0][1:]
            if [r[0] for r in raw[1:]] != current_labels:
                raise ValueError(f'Confusion matrix labels mismatch: {path}')
            if labels is None:
                labels = current_labels
                counts = [[0]*len(labels) for _ in labels]
            if current_labels != labels:
                raise ValueError(f'Inconsistent class ordering: {path}')
            for i, row in enumerate(raw[1:]):
                for j, value in enumerate(row[1:]):
                    counts[i][j] += int(float(value))
        recall = [counts[i][i]/sum(counts[i]) if sum(counts[i]) else 0 for i in range(len(labels))]
        pairs = sorted([(counts[i][j],labels[i],labels[j]) for i in range(len(labels)) for j in range(len(labels)) if i != j], reverse=True)[:5]
        worst = min(range(len(labels)), key=lambda i: recall[i])
        best = max(range(len(labels)), key=lambda i: recall[i])
        rows = [[labels[i], counts[i][i], sum(counts[i]), f'{100*recall[i]:.2f}%'] for i in range(len(labels))]
        parts.append(f'**{opt.upper()} ({len(matrices)} seeds):** Highest pooled recall: {labels[best]} ({100*recall[best]:.2f}%); lowest: {labels[worst]} ({100*recall[worst]:.2f}%). Most frequent directed error: {pairs[0][1]} → {pairs[0][2]} ({pairs[0][0]} predictions across seeds).\n\n<details><summary>All class recalls and top five directed confusions</summary>\n\n' + md_table(['Class','Correct','Total predictions for true class','Pooled recall'], rows) + '\n\n' + md_table(['True class','Predicted class','Count'], [[a,b,c] for c,a,b in pairs]) + '\n\n</details>')
    if not parts:
        return 'No matching confusion-matrix CSV files were found.'
    return 'The following counts sum predictions across independently trained seeds on the **same** test images; they do not represent additional unique test samples. Pooled recall is calculated from the summed counts.\n\n'+'\n\n'.join(parts)


def matrix_caption(figure):
    path = REPORTS / (figure.stem + '.csv')
    generic = 'Independent test confusion matrix: diagonal counts are correct predictions; off-diagonal counts are directed class errors.'
    if not path.exists():
        return generic + ' The corresponding numerical CSV was not found.'
    with path.open(newline='', encoding='utf-8-sig') as f:
        raw = list(csv.reader(f))
    if not raw:
        return generic
    labels = raw[0][1:]
    matrix = [[int(float(v)) for v in row[1:]] for row in raw[1:]]
    total = sum(map(sum, matrix))
    correct = sum(matrix[i][i] for i in range(len(labels)))
    weakest = min(range(len(labels)), key=lambda i: matrix[i][i] / max(1,sum(matrix[i])))
    count, a, b = max((matrix[i][j],i,j) for i in range(len(labels)) for j in range(len(labels)) if i != j)
    return (f'Test accuracy: {100*correct/max(1,total):.2f}%. Lowest class recall: {labels[weakest]} '
            f'({100*matrix[weakest][weakest]/max(1,sum(matrix[weakest])):.2f}%). '
            f'Most frequent directed error: {labels[a]} → {labels[b]} ({count} images). '
            f'[Numerical matrix]({relative(path)}).')


def table_archive():
    parts = ['### 20.1 Experiment and summary tables']
    files = sorted(TABLES.glob('*.csv'))
    parts.append('\n\n'.join(csv_block(p) for p in files) if files else 'No experiment CSV files found.')
    parts.append('### 20.2 Per-class classification reports and confusion matrices')
    reports = sorted(REPORTS.glob('*.csv'))
    parts.append('\n\n'.join(csv_block(p) for p in reports) if reports else 'No class-level CSV files found.')
    parts.append('### 20.3 Per-run history and configuration index')
    rows = []
    for folder in sorted(HISTORIES.iterdir()) if HISTORIES.exists() else []:
        if not folder.is_dir():
            continue
        links = []
        for filename, label in [('config.json','Config'),('history.csv','Epoch history'),('summary.json','Summary')]:
            path = folder / filename
            links.append(f'[{label}]({relative(path)})' if path.exists() else '—')
        rows.append([folder.name, *links])
    parts.append(md_table(['Run','Configuration','Epoch-level metrics','Summary'], rows) if rows else 'No saved histories found.')
    return '\n\n'.join(parts), len(files), len(reports), len(rows)


def build():
    readme = ROOT / 'README.md'
    doc = readme.read_text(encoding='utf-8')
    generate_architecture_figures(ROOT)
    all_png = sorted(p for p in FIGURES.rglob('*') if p.is_file() and p.suffix.lower() == '.png')
    seen = set()
    # Architecture plots are already embedded in Sections 3 and 4.
    for p in all_png:
        if p.parent == FIGURES / 'architecture' and p.stem in ARCH_NAMES:
            seen.add(p)
    for folder, descriptions in PLOTS.items():
        paths = [p for p in all_png if p.relative_to(FIGURES).parts[0] == folder]
        order = {stem: i for i, stem in enumerate(descriptions)}
        paths.sort(key=lambda p: (order.get(p.stem, len(order)), p.name))
        blocks = []
        for p in paths:
            caption = descriptions.get(p.stem, 'Additional measured plot; consult the corresponding experiment CSV and axes.')
            detail = baseline_detail(p.stem) if folder == 'baseline' else ''
            blocks.append(figure_block(p, caption, detail))
            seen.add(p)
        doc = replace_block(doc, folder.upper()+' FIGURES', '\n\n'.join(blocks) or 'No plots were found in this experiment directory.')
    dataset_paths = [p for p in all_png if p not in seen and (p.parent == FIGURES or any('dataset' in part.lower() or 'preprocess' in part.lower() for part in p.relative_to(FIGURES).parts)) and any(k in p.stem.lower() for k in ('sample','class','normal','dataset','distribution','preprocess'))]
    blocks = []
    for p in dataset_paths:
        stem = p.stem.lower()
        caption = ('The class-distribution plot checks stratification and balance across splits.' if ('distribution' in stem or 'class' in stem) else 'The normalization plot illustrates the transformation using training-derived mean and standard deviation.' if 'normal' in stem else 'Representative Fashion-MNIST images illustrate the input resolution and clothing categories.')
        blocks.append(figure_block(p, caption))
        seen.add(p)
    doc = replace_block(doc, 'DATASET FIGURES', '\n\n'.join(blocks) or 'Dataset-specific plots were not found in the expected paths. Generate them with `python3 -m src.plots`; any unclassified plots appear in Section 20.')
    evaluation_paths = [p for p in all_png if p.relative_to(FIGURES).parts[0] == 'evaluation']
    blocks = []
    for p in evaluation_paths:
        blocks.append(figure_block(p, matrix_caption(p)))
        seen.add(p)
    doc = replace_block(doc, 'EVALUATION FIGURES', '\n\n'.join(blocks) or 'No evaluation PNGs were found.')
    doc = replace_block(doc, 'CONFUSION ANALYSIS', confusion_analysis())
    extra = [p for p in all_png if p not in seen]
    doc = replace_block(doc, 'EXTRA FIGURES', '\n\n'.join(figure_block(p,'Additional figure from the local results directory; refer to its axes and source table.') for p in extra) or 'All discovered result PNGs are included in their respective sections above.')
    archive, n_tables, n_reports, n_runs = table_archive()
    doc = replace_block(doc, 'TABLE ARCHIVE', archive)
    # Audit every discovered PNG against the actual markdown links before write.
    missing = [p for p in all_png if f']({relative(p)})' not in doc]
    if missing:
        raise RuntimeError(f'Unlinked figures: {missing}')
    readme.write_text(doc, encoding='utf-8')
    print(f'Updated {readme}')
    print(f'Embedded {len(all_png)} PNG figures, {n_tables} experiment CSV tables, {n_reports} class-level CSV tables.')
    print(f'Indexed {n_runs} run directories (including pilot, if present).')
    print('No training performed and no experimental result CSVs changed.')


if __name__ == '__main__':
    build()
