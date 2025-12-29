import * as vscode from 'vscode';
import { GeneratorPanel } from './panels/GeneratorPanel';
import { PythonBackend } from './services/PythonBackend';
import { CodeAnalyzer } from './services/CodeAnalyzer';

let backend: PythonBackend;
let analyzer: CodeAnalyzer;

export function activate(context: vscode.ExtensionContext) {
    console.log('HIP/Triton Generator is now active!');

    // Initialize services
    backend = new PythonBackend(context);
    analyzer = new CodeAnalyzer();

    // Register Generate Triton command - directly generates and inserts code
    const generateTritonCmd = vscode.commands.registerCommand(
        'hipGenerator.generateTriton',
        () => handleGenerateAndInsert(context, 'triton')
    );

    // Register Generate HIP command - directly generates and inserts code
    const generateHipCmd = vscode.commands.registerCommand(
        'hipGenerator.generateHip',
        () => handleGenerateAndInsert(context, 'hip')
    );

    // Register Optimize Triton command - optimizes existing Triton code
    const optimizeTritonCmd = vscode.commands.registerCommand(
        'hipGenerator.optimizeTriton',
        () => handleOptimizeTriton(context)
    );

    // Register Open Panel command (for advanced usage)
    const openPanelCmd = vscode.commands.registerCommand(
        'hipGenerator.openPanel',
        () => {
            const editor = vscode.window.activeTextEditor;
            const selectedCode = editor?.selection.isEmpty ? '' : editor?.document.getText(editor.selection) || '';
            GeneratorPanel.createOrShow(context.extensionUri, backend, analyzer, {
                code: selectedCode,
                backend: 'triton',
                filePath: editor?.document.uri.fsPath
            });
        }
    );

    // Register Webview Provider for sidebar
    const webviewProvider = new GeneratorWebviewProvider(context.extensionUri, backend, analyzer);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('hipGenerator.mainView', webviewProvider)
    );

    context.subscriptions.push(generateTritonCmd, generateHipCmd, optimizeTritonCmd, openPanelCmd);

    // Check API key on activation
    checkApiKey();
}

/**
 * Handle code generation and directly insert the result into the current file
 */
async function handleGenerateAndInsert(context: vscode.ExtensionContext, backend_type: 'triton' | 'hip') {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active editor found');
        return;
    }

    const selection = editor.selection;
    if (selection.isEmpty) {
        vscode.window.showErrorMessage('Please select PyTorch code to convert');
        return;
    }

    const selectedCode = editor.document.getText(selection);
    const fullDocumentCode = editor.document.getText();
    
    // Check API key
    const config = vscode.workspace.getConfiguration('hipGenerator');
    const apiKey = config.get<string>('amdApiKey');
    
    if (!apiKey) {
        const action = await vscode.window.showErrorMessage(
            'AMD API Key not configured. Please set it in settings.',
            'Open Settings'
        );
        if (action === 'Open Settings') {
            vscode.commands.executeCommand('workbench.action.openSettings', 'hipGenerator.amdApiKey');
        }
        return;
    }

    // Analyze code to infer inputs
    const analysis = analyzer.analyzeCode(fullDocumentCode);
    
    // Show progress with detailed status
    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: `Generating ${backend_type.toUpperCase()} kernel`,
        cancellable: false
    }, async (progress) => {
        progress.report({ message: 'Initializing...' });
        
        try {
            // Generate the code with progress reporting
            const result = await backend.generateWithProgress({
                code: fullDocumentCode,  // Use full document for context
                backend: backend_type,
                getInputs: analysis.getInputs,
                getInitInputs: analysis.getInitInputs,
                maxAttempts: config.get<number>('maxAttempts') || 3
            }, progress);

            if (result.success && result.code) {
                progress.report({ message: 'Inserting generated code...' });
                
                // Extract only the ModelNew class and related code (triton imports, kernels, etc.)
                const generatedCode = extractRelevantCode(result.code);
                
                // Insert at the end of the file
                const lastLine = editor.document.lineCount - 1;
                const lastChar = editor.document.lineAt(lastLine).text.length;
                const endPosition = new vscode.Position(lastLine, lastChar);
                
                // Prepare the code to insert with proper formatting
                const separator = '\n\n# ' + '='.repeat(60) + '\n';
                const header = `# Generated ${backend_type.toUpperCase()} Kernel (${new Date().toLocaleString()})\n`;
                const codeToInsert = separator + header + '# ' + '='.repeat(60) + '\n\n' + generatedCode;
                
                await editor.edit(editBuilder => {
                    editBuilder.insert(endPosition, codeToInsert);
                });
                
                // Show success message with evaluation results
                let successMsg = `✓ ${backend_type.toUpperCase()} kernel generated and inserted!`;
                if (result.evaluation) {
                    const eval_result = result.evaluation;
                    if (eval_result.accuracy_pass) {
                        successMsg += ` | Accuracy: ✓ | Speedup: ${eval_result.speedup?.toFixed(2)}x`;
                    } else {
                        successMsg += ` | Accuracy: ✗ (may need adjustment)`;
                    }
                }
                
                vscode.window.showInformationMessage(successMsg);
                
                // Scroll to the inserted code
                const newLastLine = editor.document.lineCount - 1;
                editor.revealRange(
                    new vscode.Range(endPosition, new vscode.Position(newLastLine, 0)),
                    vscode.TextEditorRevealType.InCenter
                );
                
            } else {
                // Show more detailed error information
                let errorMsg = 'Generation failed';
                if (result.evaluation) {
                    const e = result.evaluation;
                    if (!e.compile_success) {
                        errorMsg = 'Compile failed';
                    } else if (e.has_nan) {
                        errorMsg = 'Output contains NaN values';
                    } else if (e.has_inf) {
                        errorMsg = 'Output contains Infinity values';
                    } else if (!e.accuracy_pass) {
                        errorMsg = `Accuracy failed (max_diff=${e.max_diff?.toExponential(2) || 'N/A'})`;
                    }
                } else if (result.error) {
                    // Truncate long errors
                    errorMsg = result.error.length > 100 
                        ? result.error.substring(0, 100) + '...' 
                        : result.error;
                }
                
                // If we still got code, offer to insert it anyway
                if (result.code) {
                    const action = await vscode.window.showWarningMessage(
                        `${errorMsg}. Insert generated code anyway?`,
                        'Insert Anyway',
                        'Open Panel',
                        'Cancel'
                    );
                    
                    if (action === 'Insert Anyway') {
                        // Insert the code even though it failed validation
                        const generatedCode = extractRelevantCode(result.code);
                        const lastLine = editor.document.lineCount - 1;
                        const lastChar = editor.document.lineAt(lastLine).text.length;
                        const endPosition = new vscode.Position(lastLine, lastChar);
                        
                        const separator = '\n\n# ' + '='.repeat(60) + '\n';
                        const header = `# Generated ${backend_type.toUpperCase()} Kernel (${new Date().toLocaleString()})\n`;
                        const warning = `# ⚠️ WARNING: ${errorMsg}\n`;
                        const codeToInsert = separator + header + warning + '# ' + '='.repeat(60) + '\n\n' + generatedCode;
                        
                        await editor.edit(editBuilder => {
                            editBuilder.insert(endPosition, codeToInsert);
                        });
                        
                        vscode.window.showWarningMessage(`Code inserted with warning: ${errorMsg}`);
                    } else if (action === 'Open Panel') {
                        GeneratorPanel.createOrShow(context.extensionUri, backend, analyzer, {
                            code: selectedCode,
                            backend: backend_type,
                            filePath: editor.document.uri.fsPath
                        });
                    }
                } else {
                    vscode.window.showErrorMessage(
                        errorMsg,
                        'Open Panel'
                    ).then(action => {
                        if (action === 'Open Panel') {
                            GeneratorPanel.createOrShow(context.extensionUri, backend, analyzer, {
                                code: selectedCode,
                                backend: backend_type,
                                filePath: editor.document.uri.fsPath
                            });
                        }
                    });
                }
            }
        } catch (error: any) {
            vscode.window.showErrorMessage(`Error: ${error.message}`);
        }
    });
}

/**
 * Detect wrapper class names in the code
 * Returns { refClassName, currentClassName, newClassName }
 */
function detectClassNames(code: string): { refClassName: string; currentClassName: string; newClassName: string } {
    // Find all class definitions that inherit from nn.Module
    const classPattern = /class\s+(\w+)\s*\([^)]*(?:nn\.Module|Module)[^)]*\)/g;
    const classes: string[] = [];
    let match;
    while ((match = classPattern.exec(code)) !== null) {
        classes.push(match[1]);
    }
    
    // Determine class hierarchy
    // Pattern: Model -> ModelNew -> ModelNewNew -> ...
    let refClassName = 'Model';
    let currentClassName = 'ModelNew';
    
    // Find the most "New" suffixed class as current
    const newSuffixedClasses = classes.filter(c => c.includes('New'));
    if (newSuffixedClasses.length > 0) {
        // Sort by number of "New" suffixes (more = more recent)
        newSuffixedClasses.sort((a, b) => {
            const countNew = (s: string) => (s.match(/New/g) || []).length;
            return countNew(b) - countNew(a);
        });
        currentClassName = newSuffixedClasses[0];
    }
    
    // Find the original Model class (without New suffix)
    const baseClasses = classes.filter(c => !c.includes('New'));
    if (baseClasses.length > 0) {
        refClassName = baseClasses[0];
    }
    
    // New class name = current class name + "New"
    const newClassName = currentClassName + 'New';
    
    return { refClassName, currentClassName, newClassName };
}

/**
 * Handle Triton kernel optimization - takes existing Triton code and optimizes it
 * Supports triton2triton optimization with proper class naming
 */
async function handleOptimizeTriton(context: vscode.ExtensionContext) {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active editor found');
        return;
    }

    const selection = editor.selection;
    if (selection.isEmpty) {
        vscode.window.showErrorMessage('Please select Triton code to optimize');
        return;
    }

    const selectedCode = editor.document.getText(selection);
    const fullDocumentCode = editor.document.getText();
    
    // Validate that the selected code contains Triton patterns
    if (!selectedCode.includes('@triton') && !selectedCode.includes('triton.jit') && 
        !selectedCode.includes('tl.') && !selectedCode.includes('triton.language')) {
        const action = await vscode.window.showWarningMessage(
            'Selected code does not appear to be Triton code. Continue anyway?',
            'Continue',
            'Cancel'
        );
        if (action !== 'Continue') {
            return;
        }
    }
    
    // Check API key
    const config = vscode.workspace.getConfiguration('hipGenerator');
    const apiKey = config.get<string>('amdApiKey');
    
    if (!apiKey) {
        const action = await vscode.window.showErrorMessage(
            'AMD API Key not configured. Please set it in settings.',
            'Open Settings'
        );
        if (action === 'Open Settings') {
            vscode.commands.executeCommand('workbench.action.openSettings', 'hipGenerator.amdApiKey');
        }
        return;
    }

    // Detect class names from the full document
    const { refClassName, currentClassName, newClassName } = detectClassNames(fullDocumentCode);
    
    // Try to extract original PyTorch code (reference class) for evaluation
    let originalCode: string | undefined;
    let getInputs: string | undefined;
    let getInitInputs: string | undefined;
    
    // Check if file contains the reference Model class for comparison
    const refClassPattern = new RegExp(`class\\s+${refClassName}\\s*\\([^)]*\\)[\\s\\S]*?(?=class\\s+\\w+|def\\s+get_|$)`);
    const modelMatch = fullDocumentCode.match(refClassPattern);
    if (modelMatch) {
        originalCode = modelMatch[0];
        
        // Extract get_inputs and get_init_inputs
        const getInputsMatch = fullDocumentCode.match(/def\s+get_inputs\s*\(\s*\)[\s\S]*?(?=\ndef\s|\nclass\s|$)/);
        const getInitInputsMatch = fullDocumentCode.match(/def\s+get_init_inputs\s*\(\s*\)[\s\S]*?(?=\ndef\s|\nclass\s|$)/);
        
        if (getInputsMatch) {
            getInputs = getInputsMatch[0].trim();
        }
        if (getInitInputsMatch) {
            getInitInputs = getInitInputsMatch[0].trim();
        }
        
        // Also need imports for the original code
        const importLines = fullDocumentCode.split('\n')
            .filter(line => line.trim().startsWith('import ') || line.trim().startsWith('from '))
            .filter(line => !line.includes('triton'))  // Exclude triton imports for PyTorch reference
            .join('\n');
        
        originalCode = importLines + '\n\n' + originalCode;
    }

    // Show progress with detailed status
    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: `Optimizing Triton kernel (${currentClassName} → ${newClassName})`,
        cancellable: false
    }, async (progress) => {
        progress.report({ message: 'Analyzing kernel...' });
        
        try {
            // For optimize, we need the full document code (not just selection)
            // because eval needs the complete context (imports, class definitions, get_inputs, etc.)
            // But we still show which code we're optimizing from the selection
            
            // Build the full code for optimization
            let codeForOptimization = fullDocumentCode;
            
            // If selection doesn't include the full context, use full document
            if (!selectedCode.includes('class ') && !selectedCode.includes('def get_inputs')) {
                // Selection is just the kernel - add context
                codeForOptimization = fullDocumentCode;
            } else {
                codeForOptimization = selectedCode;
            }
            
            // Call backend to optimize with evaluation support
            const result = await backend.optimize({
                code: codeForOptimization,
                originalCode: originalCode,
                getInputs: getInputs,
                getInitInputs: getInitInputs,
                maxAttempts: config.get<number>('maxAttempts') || 3,
                currentClassName: currentClassName,
                newClassName: newClassName,
                refClassName: refClassName
            }, progress);

            if (result.success && result.code) {
                progress.report({ message: 'Inserting optimized code...' });
                
                // Insert at the end of the file
                const lastLine = editor.document.lineCount - 1;
                const lastChar = editor.document.lineAt(lastLine).text.length;
                const endPosition = new vscode.Position(lastLine, lastChar);
                
                // Prepare the code to insert with proper formatting
                const separator = '\n\n# ' + '='.repeat(60) + '\n';
                const header = `# Optimized Triton Kernel: ${newClassName} (${new Date().toLocaleString()})\n`;
                const classInfo = `# ${currentClassName} → ${newClassName}\n`;
                
                // Add evaluation results as comments if available
                let evalComment = '';
                if (result.evaluation) {
                    const e = result.evaluation;
                    evalComment = `# Evaluation: Compile=${e.compile_success ? '✓' : '✗'} ` +
                        `Accuracy=${e.accuracy_pass ? '✓' : '✗'} ` +
                        `Speedup=${e.speedup?.toFixed(2) || 'N/A'}x\n`;
                }
                
                const codeToInsert = separator + header + classInfo + evalComment + '# ' + '='.repeat(60) + '\n\n' + result.code;
                
                await editor.edit(editBuilder => {
                    editBuilder.insert(endPosition, codeToInsert);
                });
                
                // Show success message with evaluation details
                let successMsg = '✓ Triton kernel optimized and inserted!';
                if (result.evaluation) {
                    const eval_result = result.evaluation;
                    if (eval_result.accuracy_pass) {
                        successMsg += ` | Accuracy: ✓ | Speedup: ${eval_result.speedup?.toFixed(2)}x`;
                    } else {
                        successMsg += ` | Accuracy: ✗ (may need manual fixes)`;
                    }
                } else {
                    successMsg += ' (no evaluation reference found)';
                }
                
                vscode.window.showInformationMessage(successMsg);
                
                // Scroll to the inserted code
                const newLastLine = editor.document.lineCount - 1;
                editor.revealRange(
                    new vscode.Range(endPosition, new vscode.Position(newLastLine, 0)),
                    vscode.TextEditorRevealType.InCenter
                );
                
            } else {
                let errorMsg = result.error || 'Optimization failed';
                
                // Show more detailed error information
                if (result.evaluation) {
                    const e = result.evaluation;
                    if (!e.compile_success) {
                        errorMsg = 'Compile failed';
                    } else if (e.has_nan) {
                        errorMsg = 'Output contains NaN values';
                    } else if (e.has_inf) {
                        errorMsg = 'Output contains Infinity values';
                    } else if (!e.accuracy_pass) {
                        errorMsg = `Accuracy failed (max_diff=${e.max_diff?.toExponential(2) || 'N/A'})`;
                    }
                }
                
                // If we still got code, offer to insert it anyway
                if (result.code) {
                    const action = await vscode.window.showWarningMessage(
                        `${errorMsg}. Insert optimized code anyway?`,
                        'Insert Anyway',
                        'Cancel'
                    );
                    
                    if (action === 'Insert Anyway') {
                        const lastLine = editor.document.lineCount - 1;
                        const lastChar = editor.document.lineAt(lastLine).text.length;
                        const endPosition = new vscode.Position(lastLine, lastChar);
                        
                        const separator = '\n\n# ' + '='.repeat(60) + '\n';
                        const header = `# Optimized Triton Kernel (${new Date().toLocaleString()})\n`;
                        const warning = `# ⚠️ WARNING: ${errorMsg}\n`;
                        const codeToInsert = separator + header + warning + '# ' + '='.repeat(60) + '\n\n' + result.code;
                        
                        await editor.edit(editBuilder => {
                            editBuilder.insert(endPosition, codeToInsert);
                        });
                        
                        vscode.window.showWarningMessage(`Code inserted with warning: ${errorMsg}`);
                    }
                } else {
                    vscode.window.showErrorMessage(errorMsg);
                }
            }
        } catch (error: any) {
            vscode.window.showErrorMessage(`Error: ${error.message}`);
        }
    });
}

/**
 * Extract the relevant generated code (ModelNew class and dependencies)
 */
function extractRelevantCode(generatedCode: string): string {
    // Find imports that are not already in the original file
    const lines = generatedCode.split('\n');
    const relevantLines: string[] = [];
    let inModelNew = false;
    let braceCount = 0;
    let foundModelNew = false;
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const trimmedLine = line.trim();
        
        // Include triton imports
        if (trimmedLine.startsWith('import triton') || 
            trimmedLine.startsWith('from triton') ||
            trimmedLine.startsWith('import os') && lines[i+1]?.includes('TRITON')) {
            relevantLines.push(line);
            continue;
        }
        
        // Include os.environ settings for triton
        if (trimmedLine.startsWith('os.environ') && 
            (trimmedLine.includes('TRITON') || trimmedLine.includes('PYTORCH'))) {
            relevantLines.push(line);
            continue;
        }
        
        // Include constants like NUM_XCDS
        if (trimmedLine.match(/^[A-Z_]+\s*=\s*\d+/)) {
            relevantLines.push(line);
            continue;
        }
        
        // Include @triton decorators and kernel functions
        if (trimmedLine.startsWith('@triton')) {
            // Include the decorator and following function
            relevantLines.push(line);
            continue;
        }
        
        // Include triton kernel definitions
        if (trimmedLine.startsWith('def ') && 
            (relevantLines.length > 0 && relevantLines[relevantLines.length - 1].trim().startsWith('@triton'))) {
            relevantLines.push(line);
            continue;
        }
        
        // Track if we're inside a triton kernel function
        if (relevantLines.length > 0) {
            const lastRelevant = relevantLines[relevantLines.length - 1];
            if (lastRelevant.trim().startsWith('def ') || 
                (lastRelevant.trim() !== '' && !lastRelevant.trim().startsWith('class '))) {
                // Continue adding lines until we hit an empty line followed by non-indented code
                if (line.startsWith('    ') || line.startsWith('\t') || trimmedLine === '') {
                    relevantLines.push(line);
                    continue;
                }
            }
        }
        
        // Include wrapper functions (usually right before ModelNew)
        if (trimmedLine.startsWith('def ') && !trimmedLine.includes('get_inputs') && 
            !trimmedLine.includes('get_init_inputs') && !trimmedLine.includes('__init__')) {
            // Check if this might be a wrapper function
            let j = i + 1;
            let isWrapper = false;
            while (j < lines.length && (lines[j].startsWith('    ') || lines[j].startsWith('\t') || lines[j].trim() === '')) {
                if (lines[j].includes('_kernel[') || lines[j].includes('kernel[')) {
                    isWrapper = true;
                    break;
                }
                j++;
            }
            if (isWrapper) {
                // Include the whole function
                relevantLines.push(line);
                i++;
                while (i < lines.length && (lines[i].startsWith('    ') || lines[i].startsWith('\t') || lines[i].trim() === '')) {
                    relevantLines.push(lines[i]);
                    if (lines[i].trim() !== '' && !lines[i].startsWith('    ') && !lines[i].startsWith('\t')) {
                        break;
                    }
                    i++;
                }
                i--; // Back up one since the for loop will increment
                continue;
            }
        }
        
        // Include ModelNew class
        if (trimmedLine.startsWith('class ModelNew')) {
            foundModelNew = true;
            inModelNew = true;
            relevantLines.push(line);
            continue;
        }
        
        // If we're in ModelNew class, keep adding until the class ends
        if (inModelNew) {
            relevantLines.push(line);
            // Track class ending by indentation
            if (trimmedLine !== '' && !line.startsWith('    ') && !line.startsWith('\t') && 
                !trimmedLine.startsWith('class ') && !trimmedLine.startsWith('def ')) {
                inModelNew = false;
            }
            continue;
        }
    }
    
    // If we didn't find ModelNew, return the whole generated code
    if (!foundModelNew) {
        return generatedCode;
    }
    
    return relevantLines.join('\n');
}

function checkApiKey() {
    const config = vscode.workspace.getConfiguration('hipGenerator');
    const apiKey = config.get<string>('amdApiKey');
    
    if (!apiKey) {
        vscode.window.showInformationMessage(
            'HIP Generator: Please configure your AMD API Key in settings',
            'Open Settings'
        ).then(action => {
            if (action === 'Open Settings') {
                vscode.commands.executeCommand('workbench.action.openSettings', 'hipGenerator.amdApiKey');
            }
        });
    }
}

class GeneratorWebviewProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;

    constructor(
        private readonly _extensionUri: vscode.Uri,
        private readonly _backend: PythonBackend,
        private readonly _analyzer: CodeAnalyzer
    ) {}

    resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        // Handle messages from webview
        webviewView.webview.onDidReceiveMessage(async (message) => {
            switch (message.command) {
                case 'generate':
                    await this._handleGenerate(message);
                    break;
                case 'analyze':
                    await this._handleAnalyze(message);
                    break;
                case 'openSettings':
                    vscode.commands.executeCommand('workbench.action.openSettings', 'hipGenerator');
                    break;
                case 'openFromSelection':
                    vscode.commands.executeCommand(
                        message.backend === 'triton' ? 'hipGenerator.generateTriton' : 'hipGenerator.generateHip'
                    );
                    break;
            }
        });
    }

    private async _handleGenerate(message: any) {
        if (!this._view) return;

        try {
            this._view.webview.postMessage({ command: 'status', status: 'generating' });

            const result = await this._backend.generate({
                code: message.code,
                backend: message.backend,
                getInputs: message.getInputs,
                getInitInputs: message.getInitInputs,
                maxAttempts: message.maxAttempts || 3
            });

            this._view.webview.postMessage({ 
                command: 'result', 
                result: result 
            });
        } catch (error: any) {
            this._view.webview.postMessage({ 
                command: 'error', 
                error: error.message 
            });
        }
    }

    private async _handleAnalyze(message: any) {
        if (!this._view) return;

        try {
            const analysis = this._analyzer.analyzeCode(message.code);
            this._view.webview.postMessage({ 
                command: 'analysis', 
                analysis: analysis 
            });
        } catch (error: any) {
            this._view.webview.postMessage({ 
                command: 'error', 
                error: error.message 
            });
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HIP Generator</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--vscode-foreground);
            background-color: var(--vscode-sideBar-background);
            padding: 12px;
            margin: 0;
        }
        .section {
            margin-bottom: 16px;
        }
        .section-title {
            font-weight: bold;
            margin-bottom: 8px;
            color: var(--vscode-titleBar-activeForeground);
        }
        button {
            background-color: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 8px 16px;
            cursor: pointer;
            border-radius: 2px;
            width: 100%;
            margin-bottom: 8px;
        }
        button:hover {
            background-color: var(--vscode-button-hoverBackground);
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .info {
            font-size: 12px;
            color: var(--vscode-descriptionForeground);
            margin-top: 8px;
        }
        .status {
            padding: 8px;
            border-radius: 4px;
            margin-top: 8px;
        }
        .status.generating {
            background-color: var(--vscode-inputValidation-infoBackground);
            border: 1px solid var(--vscode-inputValidation-infoBorder);
        }
        .status.success {
            background-color: var(--vscode-inputValidation-infoBackground);
            border: 1px solid var(--vscode-charts-green);
        }
        .status.error {
            background-color: var(--vscode-inputValidation-errorBackground);
            border: 1px solid var(--vscode-inputValidation-errorBorder);
        }
    </style>
</head>
<body>
    <div class="section">
        <div class="section-title">Quick Actions</div>
        <button onclick="openFromSelection('triton')">
            🚀 Generate Triton from Selection
        </button>
        <button onclick="openFromSelection('hip')">
            ⚡ Generate HIP from Selection
        </button>
    </div>
    
    <div class="section">
        <div class="section-title">Settings</div>
        <button onclick="openSettings()">
            ⚙️ Configure API Key & Options
        </button>
    </div>
    
    <div class="info">
        <p><strong>Usage:</strong></p>
        <ol style="padding-left: 20px; margin: 8px 0;">
            <li>Select PyTorch code in editor</li>
            <li>Right-click → "Generate Triton/HIP Kernel"</li>
            <li>Generated code will be inserted at the end of the file</li>
        </ol>
    </div>

    <div id="status"></div>

    <script>
        const vscode = acquireVsCodeApi();
        
        function openFromSelection(backend) {
            vscode.postMessage({ 
                command: 'openFromSelection',
                backend: backend
            });
        }
        
        function openSettings() {
            vscode.postMessage({ command: 'openSettings' });
        }

        window.addEventListener('message', event => {
            const message = event.data;
            const statusEl = document.getElementById('status');
            
            switch (message.command) {
                case 'status':
                    statusEl.className = 'status ' + message.status;
                    statusEl.textContent = message.status === 'generating' 
                        ? '⏳ Generating kernel...' 
                        : message.status;
                    break;
                case 'error':
                    statusEl.className = 'status error';
                    statusEl.textContent = '❌ ' + message.error;
                    break;
            }
        });
    </script>
</body>
</html>`;
    }
}

export function deactivate() {
    if (backend) {
        backend.dispose();
    }
}
