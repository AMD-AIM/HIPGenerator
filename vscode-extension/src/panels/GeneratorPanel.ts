import * as vscode from 'vscode';
import { PythonBackend, GenerationResult } from '../services/PythonBackend';
import { CodeAnalyzer, CodeAnalysis } from '../services/CodeAnalyzer';

interface InitialData {
    code: string;
    backend: 'triton' | 'hip';
    filePath?: string;
}

export class GeneratorPanel {
    public static currentPanel: GeneratorPanel | undefined;
    private readonly _panel: vscode.WebviewPanel;
    private readonly _extensionUri: vscode.Uri;
    private _disposables: vscode.Disposable[] = [];
    private _backend: PythonBackend;
    private _analyzer: CodeAnalyzer;

    public static createOrShow(
        extensionUri: vscode.Uri,
        backend: PythonBackend,
        analyzer: CodeAnalyzer,
        initialData?: InitialData
    ) {
        const column = vscode.ViewColumn.Beside;

        if (GeneratorPanel.currentPanel) {
            GeneratorPanel.currentPanel._panel.reveal(column);
            if (initialData) {
                GeneratorPanel.currentPanel._initWithData(initialData);
            }
            return;
        }

        const panel = vscode.window.createWebviewPanel(
            'hipGenerator',
            'HIP/Triton Generator',
            column,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [extensionUri]
            }
        );

        GeneratorPanel.currentPanel = new GeneratorPanel(panel, extensionUri, backend, analyzer);
        
        if (initialData) {
            // Wait for webview to be ready, then send data
            setTimeout(() => {
                GeneratorPanel.currentPanel?._initWithData(initialData);
            }, 500);
        }
    }

    private constructor(
        panel: vscode.WebviewPanel,
        extensionUri: vscode.Uri,
        backend: PythonBackend,
        analyzer: CodeAnalyzer
    ) {
        this._panel = panel;
        this._extensionUri = extensionUri;
        this._backend = backend;
        this._analyzer = analyzer;

        this._panel.webview.html = this._getHtmlForWebview();

        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);

        this._panel.webview.onDidReceiveMessage(
            async (message) => {
                await this._handleMessage(message);
            },
            null,
            this._disposables
        );
    }

    private _initWithData(data: InitialData) {
        // First analyze the code
        const analysis = this._analyzer.analyzeCode(data.code);
        
        this._panel.webview.postMessage({
            command: 'init',
            code: data.code,
            backend: data.backend,
            analysis: analysis
        });
    }

    private async _handleMessage(message: any) {
        switch (message.command) {
            case 'analyze':
                const analysis = this._analyzer.analyzeCode(message.code);
                this._panel.webview.postMessage({
                    command: 'analysis',
                    analysis: analysis
                });
                break;

            case 'generate':
                await this._handleGenerate(message);
                break;

            case 'evaluate':
                await this._handleEvaluate(message);
                break;

            case 'saveCode':
                await this._handleSaveCode(message);
                break;

            case 'openSettings':
                vscode.commands.executeCommand('workbench.action.openSettings', 'hipGenerator');
                break;
        }
    }

    private async _handleGenerate(message: any) {
        this._panel.webview.postMessage({
            command: 'status',
            status: 'generating',
            message: 'Generating kernel code...'
        });

        try {
            const result = await this._backend.generate({
                code: message.code,
                backend: message.backend,
                getInputs: message.getInputs,
                getInitInputs: message.getInitInputs,
                maxAttempts: message.maxAttempts || 1
            });

            this._panel.webview.postMessage({
                command: 'generationResult',
                result: result
            });
        } catch (error: any) {
            this._panel.webview.postMessage({
                command: 'error',
                error: error.message
            });
        }
    }

    private async _handleEvaluate(message: any) {
        this._panel.webview.postMessage({
            command: 'status',
            status: 'evaluating',
            message: 'Evaluating generated code...'
        });

        try {
            const result = await this._backend.evaluate({
                generatedCode: message.generatedCode,
                originalCode: message.originalCode,
                getInputs: message.getInputs,
                getInitInputs: message.getInitInputs,
                backend: message.backend
            });

            this._panel.webview.postMessage({
                command: 'evaluationResult',
                result: result
            });
        } catch (error: any) {
            this._panel.webview.postMessage({
                command: 'error',
                error: error.message
            });
        }
    }

    private async _handleSaveCode(message: any) {
        const uri = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.file('generated_kernel.py'),
            filters: {
                'Python': ['py']
            }
        });

        if (uri) {
            const encoder = new TextEncoder();
            await vscode.workspace.fs.writeFile(uri, encoder.encode(message.code));
            vscode.window.showInformationMessage(`Saved to ${uri.fsPath}`);
        }
    }

    public dispose() {
        GeneratorPanel.currentPanel = undefined;

        this._panel.dispose();

        while (this._disposables.length) {
            const disposable = this._disposables.pop();
            if (disposable) {
                disposable.dispose();
            }
        }
    }

    private _getHtmlForWebview(): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HIP/Triton Generator</title>
    <style>
        :root {
            --bg-primary: var(--vscode-editor-background);
            --bg-secondary: var(--vscode-sideBar-background);
            --text-primary: var(--vscode-foreground);
            --text-secondary: var(--vscode-descriptionForeground);
            --accent: var(--vscode-button-background);
            --accent-hover: var(--vscode-button-hoverBackground);
            --border: var(--vscode-panel-border);
            --success: var(--vscode-charts-green, #4caf50);
            --error: var(--vscode-errorForeground, #f44336);
            --warning: var(--vscode-editorWarning-foreground, #ff9800);
        }
        
        * {
            box-sizing: border-box;
        }
        
        body {
            font-family: var(--vscode-font-family);
            font-size: 13px;
            color: var(--text-primary);
            background-color: var(--bg-primary);
            margin: 0;
            padding: 0;
            height: 100vh;
            overflow: hidden;
        }
        
        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: auto 1fr auto;
            height: 100vh;
            gap: 1px;
            background-color: var(--border);
        }
        
        .header {
            grid-column: 1 / -1;
            background-color: var(--bg-secondary);
            padding: 12px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border);
        }
        
        .header h1 {
            margin: 0;
            font-size: 16px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .header-actions {
            display: flex;
            gap: 8px;
        }
        
        .panel {
            background-color: var(--bg-primary);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .panel-header {
            padding: 8px 12px;
            background-color: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .panel-content {
            flex: 1;
            overflow: auto;
            padding: 12px;
        }
        
        .code-area {
            width: 100%;
            height: 100%;
            min-height: 200px;
            background-color: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            font-family: var(--vscode-editor-font-family, 'Consolas', monospace);
            font-size: 12px;
            padding: 8px;
            resize: none;
            border-radius: 4px;
        }
        
        .code-area:focus {
            outline: 1px solid var(--accent);
        }
        
        .footer {
            grid-column: 1 / -1;
            background-color: var(--bg-secondary);
            padding: 12px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-top: 1px solid var(--border);
        }
        
        .btn {
            background-color: var(--accent);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 8px 16px;
            cursor: pointer;
            border-radius: 4px;
            font-size: 13px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: background-color 0.2s;
        }
        
        .btn:hover:not(:disabled) {
            background-color: var(--accent-hover);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .btn-secondary {
            background-color: transparent;
            border: 1px solid var(--border);
            color: var(--text-primary);
        }
        
        .btn-secondary:hover:not(:disabled) {
            background-color: var(--bg-secondary);
        }
        
        .btn-success {
            background-color: var(--success);
        }
        
        .form-group {
            margin-bottom: 16px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 6px;
            font-weight: 500;
        }
        
        .form-group input,
        .form-group select {
            width: 100%;
            padding: 8px;
            background-color: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            border-radius: 4px;
        }
        
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        
        .status-bar {
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--text-secondary);
        }
        
        .status-bar.generating {
            color: var(--accent);
        }
        
        .status-bar.success {
            color: var(--success);
        }
        
        .status-bar.error {
            color: var(--error);
        }
        
        .spinner {
            width: 16px;
            height: 16px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 12px;
        }
        
        .result-card {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
        }
        
        .result-card h4 {
            margin: 0 0 8px 0;
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
        }
        
        .result-card .value {
            font-size: 20px;
            font-weight: 600;
        }
        
        .result-card.success .value {
            color: var(--success);
        }
        
        .result-card.error .value {
            color: var(--error);
        }
        
        .tabs {
            display: flex;
            border-bottom: 1px solid var(--border);
            background-color: var(--bg-secondary);
        }
        
        .tab {
            padding: 8px 16px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            color: var(--text-secondary);
            transition: all 0.2s;
        }
        
        .tab:hover {
            color: var(--text-primary);
        }
        
        .tab.active {
            color: var(--text-primary);
            border-bottom-color: var(--accent);
        }
        
        .help-text {
            font-size: 11px;
            color: var(--text-secondary);
            margin-top: 4px;
        }
        
        .badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 500;
        }
        
        .badge-triton {
            background-color: #667eea33;
            color: #667eea;
        }
        
        .badge-hip {
            background-color: #ed640033;
            color: #ed6400;
        }
        
        .collapsible {
            cursor: pointer;
            user-select: none;
        }
        
        .collapsible::before {
            content: '▶';
            display: inline-block;
            margin-right: 6px;
            transition: transform 0.2s;
            font-size: 10px;
        }
        
        .collapsible.open::before {
            transform: rotate(90deg);
        }
        
        .collapsible-content {
            display: none;
            padding-top: 8px;
        }
        
        .collapsible-content.open {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>
                <span>⚡</span>
                HIP/Triton Generator
            </h1>
            <div class="header-actions">
                <select id="backendSelect" onchange="updateBackend()">
                    <option value="triton">🔷 Triton</option>
                    <option value="hip">🔶 HIP</option>
                </select>
                <button class="btn btn-secondary" onclick="openSettings()">
                    ⚙️ Settings
                </button>
            </div>
        </header>
        
        <div class="panel" id="inputPanel">
            <div class="panel-header">
                <span>📝 Input PyTorch Code</span>
                <button class="btn btn-secondary" onclick="analyzeCode()" style="padding: 4px 8px; font-size: 11px;">
                    🔍 Analyze
                </button>
            </div>
            <div class="panel-content">
                <textarea id="inputCode" class="code-area" placeholder="Paste your PyTorch code here...

Example:
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        return torch.matmul(x, y)

def get_inputs():
    return [torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda'),
            torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')]

def get_init_inputs():
    return []
"></textarea>
            </div>
        </div>
        
        <div class="panel" id="outputPanel">
            <div class="tabs">
                <div class="tab active" onclick="showTab('generated')">Generated Code</div>
                <div class="tab" onclick="showTab('config')">Configuration</div>
                <div class="tab" onclick="showTab('results')">Results</div>
            </div>
            <div class="panel-content">
                <div id="generatedTab" class="tab-content">
                    <textarea id="outputCode" class="code-area" readonly placeholder="Generated kernel code will appear here..."></textarea>
                </div>
                
                <div id="configTab" class="tab-content" style="display: none;">
                    <div class="form-group">
                        <label class="collapsible" onclick="toggleCollapsible(this)">
                            get_inputs() - Test Input Generator
                        </label>
                        <div class="collapsible-content">
                            <textarea id="getInputs" class="code-area" style="height: 120px;" placeholder="def get_inputs():
    return [torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')]"></textarea>
                            <div class="help-text">
                                Function that returns a list of input tensors for testing.
                                Will be auto-inferred from your code if possible.
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label class="collapsible" onclick="toggleCollapsible(this)">
                            get_init_inputs() - Model Init Arguments
                        </label>
                        <div class="collapsible-content">
                            <textarea id="getInitInputs" class="code-area" style="height: 80px;" placeholder="def get_init_inputs():
    return []"></textarea>
                            <div class="help-text">
                                Function that returns arguments for Model.__init__().
                                Usually empty list for simple models.
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label>Max Attempts</label>
                            <input type="number" id="maxAttempts" value="3" min="1" max="10">
                        </div>
                        <div class="form-group">
                            <label>Target Speedup</label>
                            <input type="number" id="targetSpeedup" value="1.0" min="0.1" step="0.1">
                        </div>
                    </div>
                </div>
                
                <div id="resultsTab" class="tab-content" style="display: none;">
                    <div id="resultsContent">
                        <p style="color: var(--text-secondary);">
                            No results yet. Generate and evaluate code to see results.
                        </p>
                    </div>
                </div>
            </div>
        </div>
        
        <footer class="footer">
            <div class="status-bar" id="statusBar">
                <span>Ready</span>
            </div>
            <div style="display: flex; gap: 8px;">
                <button class="btn btn-secondary" onclick="saveCode()" id="saveBtn" disabled>
                    💾 Save
                </button>
                <button class="btn btn-secondary" onclick="evaluateCode()" id="evalBtn" disabled>
                    🧪 Evaluate
                </button>
                <button class="btn" onclick="generateCode()" id="generateBtn">
                    🚀 Generate
                </button>
            </div>
        </footer>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        
        let currentBackend = 'triton';
        let analysisResult = null;
        let generatedCode = '';
        const TARGET_SPEEDUP = ${vscode.workspace.getConfiguration('hipGenerator').get<number>('targetSpeedup') || 1.0};
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            // Load saved state if any
            const state = vscode.getState();
            if (state) {
                if (state.inputCode) document.getElementById('inputCode').value = state.inputCode;
                if (state.outputCode) {
                    document.getElementById('outputCode').value = state.outputCode;
                    generatedCode = state.outputCode;
                }
                if (state.backend) {
                    currentBackend = state.backend;
                    document.getElementById('backendSelect').value = state.backend;
                }
            }
        });
        
        // Handle messages from extension
        window.addEventListener('message', event => {
            const message = event.data;
            
            switch (message.command) {
                case 'init':
                    document.getElementById('inputCode').value = message.code;
                    document.getElementById('backendSelect').value = message.backend;
                    currentBackend = message.backend;
                    if (message.analysis) {
                        applyAnalysis(message.analysis);
                    }
                    saveState();
                    break;
                    
                case 'analysis':
                    applyAnalysis(message.analysis);
                    break;
                    
                case 'status':
                    updateStatus(message.status, message.message);
                    break;
                    
                case 'generationResult':
                    handleGenerationResult(message.result);
                    break;
                    
                case 'evaluationResult':
                    handleEvaluationResult(message.result);
                    break;
                    
                case 'error':
                    showError(message.error);
                    break;
            }
        });
        
        function applyAnalysis(analysis) {
            analysisResult = analysis;
            
            // Auto-fill get_inputs
            if (analysis.getInputs) {
                document.getElementById('getInputs').value = analysis.getInputs;
            }
            
            // Auto-fill get_init_inputs
            if (analysis.getInitInputs) {
                document.getElementById('getInitInputs').value = analysis.getInitInputs;
            }
            
            // Expand the collapsibles to show the inferred values
            document.querySelectorAll('.collapsible').forEach(el => {
                el.classList.add('open');
                el.nextElementSibling.classList.add('open');
            });
            
            // Switch to config tab to show inferred inputs
            showTab('config');
        }
        
        function updateBackend() {
            currentBackend = document.getElementById('backendSelect').value;
            saveState();
        }
        
        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
            
            event.target.classList.add('active');
            document.getElementById(tabName + 'Tab').style.display = 'block';
        }
        
        function toggleCollapsible(el) {
            el.classList.toggle('open');
            el.nextElementSibling.classList.toggle('open');
        }
        
        function analyzeCode() {
            const code = document.getElementById('inputCode').value;
            if (!code.trim()) {
                showError('Please enter PyTorch code to analyze');
                return;
            }
            
            vscode.postMessage({
                command: 'analyze',
                code: code
            });
        }
        
        function generateCode() {
            const code = document.getElementById('inputCode').value;
            if (!code.trim()) {
                showError('Please enter PyTorch code to convert');
                return;
            }
            
            const getInputs = document.getElementById('getInputs').value;
            const getInitInputs = document.getElementById('getInitInputs').value;
            const maxAttempts = parseInt(document.getElementById('maxAttempts').value) || 3;
            
            document.getElementById('generateBtn').disabled = true;
            
            vscode.postMessage({
                command: 'generate',
                code: code,
                backend: currentBackend,
                getInputs: getInputs,
                getInitInputs: getInitInputs,
                maxAttempts: maxAttempts
            });
        }
        
        function evaluateCode() {
            if (!generatedCode) {
                showError('No generated code to evaluate');
                return;
            }
            
            const originalCode = document.getElementById('inputCode').value;
            const getInputs = document.getElementById('getInputs').value;
            const getInitInputs = document.getElementById('getInitInputs').value;
            
            document.getElementById('evalBtn').disabled = true;
            
            vscode.postMessage({
                command: 'evaluate',
                generatedCode: generatedCode,
                originalCode: originalCode,
                getInputs: getInputs,
                getInitInputs: getInitInputs,
                backend: currentBackend
            });
        }
        
        function saveCode() {
            if (!generatedCode) {
                showError('No code to save');
                return;
            }
            
            vscode.postMessage({
                command: 'saveCode',
                code: generatedCode
            });
        }
        
        function openSettings() {
            vscode.postMessage({ command: 'openSettings' });
        }
        
        function updateStatus(status, message) {
            const statusBar = document.getElementById('statusBar');
            statusBar.className = 'status-bar ' + status;
            
            if (status === 'generating' || status === 'evaluating') {
                statusBar.innerHTML = '<div class="spinner"></div><span>' + (message || status) + '</span>';
            } else {
                statusBar.innerHTML = '<span>' + (message || status) + '</span>';
            }
        }
        
        function handleGenerationResult(result) {
            document.getElementById('generateBtn').disabled = false;
            
            if (result.success) {
                generatedCode = result.code;
                document.getElementById('outputCode').value = result.code;
                document.getElementById('saveBtn').disabled = false;
                document.getElementById('evalBtn').disabled = false;
                
                updateStatus('success', '✓ Generated successfully');
                showTab('generated');
                
                // Show results if available
                if (result.evaluation) {
                    showResults(result.evaluation);
                }
            } else {
                updateStatus('error', '✗ Generation failed');
                showError(result.error || 'Unknown error');
            }
            
            saveState();
        }
        
        function handleEvaluationResult(result) {
            document.getElementById('evalBtn').disabled = false;
            
            showResults(result);
            showTab('results');
            
            if (result.accuracy_pass) {
                updateStatus('success', '✓ Evaluation passed - ' + result.speedup.toFixed(2) + 'x speedup');
            } else {
                updateStatus('error', '✗ Evaluation failed');
            }
        }
        
        function showResults(result) {
            const resultsContent = document.getElementById('resultsContent');
            
            const compileClass = result.compile_success ? 'success' : 'error';
            const accuracyClass = result.accuracy_pass ? 'success' : 'error';
            const speedupClass = result.speedup >= TARGET_SPEEDUP ? 'success' : (result.speedup > 0 ? '' : 'error');
            
            resultsContent.innerHTML = \`
                <div class="results-grid">
                    <div class="result-card \${compileClass}">
                        <h4>Compile</h4>
                        <div class="value">\${result.compile_success ? '✓' : '✗'}</div>
                    </div>
                    <div class="result-card \${accuracyClass}">
                        <h4>Accuracy</h4>
                        <div class="value">\${result.accuracy_pass ? '✓' : '✗'}</div>
                    </div>
                    <div class="result-card \${speedupClass}">
                        <h4>Speedup</h4>
                        <div class="value">\${result.speedup ? result.speedup.toFixed(2) + 'x' : 'N/A'}</div>
                    </div>
                </div>
                
                <div style="margin-top: 16px;">
                    <h4 style="margin-bottom: 8px;">Details</h4>
                    <table style="width: 100%; font-size: 12px;">
                        <tr><td>Reference Time:</td><td>\${result.ref_time_ms?.toFixed(3) || 'N/A'} ms</td></tr>
                        <tr><td>Generated Time:</td><td>\${result.new_time_ms?.toFixed(3) || 'N/A'} ms</td></tr>
                        <tr><td>Max Diff:</td><td>\${result.max_diff?.toExponential(2) || 'N/A'}</td></tr>
                        <tr><td>Mean Diff:</td><td>\${result.mean_diff?.toExponential(2) || 'N/A'}</td></tr>
                    </table>
                </div>
                
                \${result.error ? \`
                    <div style="margin-top: 16px; padding: 12px; background: var(--vscode-inputValidation-errorBackground); border-radius: 4px;">
                        <h4 style="color: var(--error); margin-bottom: 8px;">Error</h4>
                        <pre style="font-size: 11px; overflow: auto; max-height: 200px;">\${escapeHtml(result.error)}</pre>
                    </div>
                \` : ''}
            \`;
        }
        
        function showError(message) {
            updateStatus('error', '✗ ' + message);
            document.getElementById('generateBtn').disabled = false;
            document.getElementById('evalBtn').disabled = false;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function saveState() {
            vscode.setState({
                inputCode: document.getElementById('inputCode').value,
                outputCode: document.getElementById('outputCode').value,
                backend: currentBackend
            });
        }
        
        // Auto-save on input change
        document.getElementById('inputCode').addEventListener('input', saveState);
    </script>
</body>
</html>`;
    }
}











