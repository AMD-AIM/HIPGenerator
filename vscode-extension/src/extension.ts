import * as vscode from 'vscode';
import { GeneratorPanel } from './panels/GeneratorPanel';
import { PythonBackend, GenerationResult } from './services/PythonBackend';
import { CodeAnalyzer } from './services/CodeAnalyzer';

let backend: PythonBackend;
let analyzer: CodeAnalyzer;

// VERSION: 1.0.0
const EXTENSION_VERSION = '1.0.0';

export function activate(context: vscode.ExtensionContext) {
    console.log(`HIP/Triton Generator v${EXTENSION_VERSION} is now active!`);
    vscode.window.showInformationMessage(`HIP Generator loaded: ${EXTENSION_VERSION}`);

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
    webviewProvider = new GeneratorWebviewProvider(context.extensionUri, backend, analyzer);
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
    
    // Create task ID for history tracking
    const taskId = `gen-${Date.now()}`;
    const fileName = editor.document.fileName.split('/').pop() || editor.document.fileName.split('\\').pop();
    
    // Update sidebar with pending task
    if (webviewProvider) {
        webviewProvider.setCurrentTask(taskId);
        webviewProvider.updateHistory({
            id: taskId,
            timestamp: new Date(),
            backend: backend_type,
            status: 'generating',
            fileName: fileName,
            attempt: 1,
            maxAttempts: config.get<number>('maxAttempts') || 3
        });
    }
    
    // Show progress with detailed status
    const result = await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: `Generating ${backend_type.toUpperCase()} kernel`,
        cancellable: true
    }, async (progress, token): Promise<GenerationResult> => {
        progress.report({ message: 'Initializing...' });
        
        // Check for cancellation
        token.onCancellationRequested(() => {
            if (webviewProvider) {
                webviewProvider.updateHistory({
                    id: taskId,
                    timestamp: new Date(),
                    backend: backend_type,
                    status: 'cancelled',
                    fileName: fileName
                });
                webviewProvider.setCurrentTask(null);
            }
        });
        
        try {
            // Generate the code with progress reporting
            return await backend.generateWithProgress({
                code: fullDocumentCode,  // Use full document for context
                backend: backend_type,
                getInputs: analysis.getInputs,
                getInitInputs: analysis.getInitInputs,
                maxAttempts: config.get<number>('maxAttempts') || 3
            }, progress, (attempt, maxAttempts) => {
                // Update sidebar with current attempt
                if (webviewProvider) {
                    webviewProvider.updateHistory({
                        id: taskId,
                        timestamp: new Date(),
                        backend: backend_type,
                        status: 'generating',
                        fileName: fileName,
                        attempt: attempt,
                        maxAttempts: maxAttempts
                    });
                }
            });
        } catch (error: any) {
            return { success: false, error: error.message };
        }
    });
    
    // Clear current task
    if (webviewProvider) {
        webviewProvider.setCurrentTask(null);
    }

    // Handle result OUTSIDE of withProgress to avoid blocking issues
    // Debug log
    // Log to file for debugging
    const fs = require('fs');
    const logFile = '/tmp/hip_generator_debug.log';
    const logMsg = (msg: string) => {
        const line = `[${new Date().toISOString()}] ${msg}\n`;
        fs.appendFileSync(logFile, line);
    };
    
    logMsg(`=== Generate Triton Result ===`);
    logMsg(`success=${result.success}, hasCode=${!!result.code}`);
    logMsg(`speedup=${result.evaluation?.speedup}, accuracy=${result.evaluation?.accuracy_pass}`);
    logMsg(`targetSpeedup config=${config.get<number>('targetSpeedup')}`);
    
    // Update sidebar history with final result
    const speedupForHistory = result.evaluation?.speedup || 0;
    const historyEntry: GenerationHistoryEntry = {
        id: taskId,
        timestamp: new Date(),
        backend: backend_type,
        status: result.success ? 'completed' : 'failed',
        fileName: fileName,
        evaluation: result.evaluation ? {
            compiled: result.evaluation.compile_success || false,
            correctness: result.evaluation.accuracy_pass || false,
            speedup: speedupForHistory,
            runtime: result.evaluation.new_time_ms,
            refRuntime: result.evaluation.ref_time_ms
        } : undefined,
        error: result.error,
        code: result.code
    };
    if (webviewProvider) {
        webviewProvider.updateHistory(historyEntry);
    }
    
    if (result.success && result.code) {
        // Extract only the ModelNew class and related code (triton imports, kernels, etc.)
        const generatedCode = extractRelevantCode(result.code);
        
        // Check speedup against target
        const speedup = result.evaluation?.speedup || 0;
        const accuracyPass = result.evaluation?.accuracy_pass ?? false;
        const targetSpeedup = config.get<number>('targetSpeedup') || 1.0;
        
        logMsg(`Parsed: speedup=${speedup}, accuracyPass=${accuracyPass}, targetSpeedup=${targetSpeedup}`);
        logMsg(`shouldAutoInsert = ${speedup} >= ${targetSpeedup} && ${accuracyPass} = ${speedup >= targetSpeedup && accuracyPass}`);
        
        const insertCode = async () => {
            const lastLine = editor.document.lineCount - 1;
            const lastChar = editor.document.lineAt(lastLine).text.length;
            const endPosition = new vscode.Position(lastLine, lastChar);
            
            // Prepare the code to insert with proper formatting
            const separator = '\n\n# ' + '='.repeat(60) + '\n';
            const header = `# Generated ${backend_type.toUpperCase()} Kernel (${new Date().toLocaleString()})\n`;
            const speedupInfo = speedup >= targetSpeedup 
                ? `# Evaluation: Accuracy=✓ Speedup=${speedup.toFixed(2)}x\n` 
                : `# ⚠ WARNING: Speedup=${speedup.toFixed(2)}x (below target ${targetSpeedup}x)\n`;
            const codeToInsert = separator + header + speedupInfo + '# ' + '='.repeat(60) + '\n\n' + generatedCode;
            
            await editor.edit(editBuilder => {
                editBuilder.insert(endPosition, codeToInsert);
            });
            
            // Scroll to the inserted code
            const newLastLine = editor.document.lineCount - 1;
            editor.revealRange(
                new vscode.Range(endPosition, new vscode.Position(newLastLine, 0)),
                vscode.TextEditorRevealType.InCenter
            );
        };
        
        // Decision logic: only auto-insert if speedup >= target AND accuracy passes
        const shouldAutoInsert = speedup >= targetSpeedup && accuracyPass;
        
        logMsg(`Decision: shouldAutoInsert=${shouldAutoInsert}`);
        
        if (shouldAutoInsert) {
            // Good performance - auto insert
            logMsg(`AUTO-INSERTING (speedup >= target)`);
            await insertCode();
            vscode.window.showInformationMessage(
                `✓ ${backend_type.toUpperCase()} kernel generated! Speedup: ${speedup.toFixed(2)}x`
            );
        } else {
            // Below target OR accuracy failed - MUST ask user with modal dialog
            logMsg(`Showing MODAL dialog (speedup < target or accuracy failed)`);
            const action = await vscode.window.showWarningMessage(
                `⚠️ Performance: ${speedup.toFixed(2)}x (target: ${targetSpeedup}x), Accuracy: ${accuracyPass ? 'PASS' : 'FAIL'}`,
                { modal: true },  // Force modal dialog - user MUST respond
                'Insert Code',
                'Cancel'
            );
            
            logMsg(`User action: ${action || 'dismissed/cancelled'}`);
            
            if (action === 'Insert Code') {
                logMsg(`USER CLICKED INSERT - inserting code`);
                await insertCode();
            } else {
                logMsg(`User cancelled/dismissed - NOT inserting`);
            }
        }
        
    } else {
        logMsg(`=== ELSE BRANCH: result.success=false ===`);
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
        
        logMsg(`errorMsg=${errorMsg}, hasCode=${!!result.code}`);
        
        // If we still got code, offer to insert it anyway
        if (result.code) {
            logMsg(`Showing error dialog with Insert Anyway option`);
            const action = await vscode.window.showWarningMessage(
                `${errorMsg}. Insert generated code anyway?`,
                'Insert Anyway',
                'Open Panel',
                'Cancel'
            );
            
            logMsg(`User action in error dialog: ${action || 'dismissed'}`);
            
            if (action === 'Insert Anyway') {
                logMsg(`USER CLICKED INSERT ANYWAY in error branch - inserting`);
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
    // For triton2triton: the user's Triton class (e.g., ModelNew) is the BASELINE
    const { currentClassName: baselineClassName } = detectClassNames(fullDocumentCode);
    
    // Generate a unique optimized class name that doesn't conflict with baseline
    const optimizedClassName = baselineClassName + 'Optimized';

    // Create task ID for history tracking
    const taskId = `opt-${Date.now()}`;
    const fileName = editor.document.fileName.split('/').pop() || editor.document.fileName.split('\\').pop();
    
    // Update sidebar with pending task
    if (webviewProvider) {
        webviewProvider.setCurrentTask(taskId);
        webviewProvider.updateHistory({
            id: taskId,
            timestamp: new Date(),
            backend: 'triton',
            status: 'generating',
            fileName: `${fileName} (optimize)`,
            attempt: 1,
            maxAttempts: config.get<number>('maxAttempts') || 3
        });
    }

    // Show progress with detailed status
    const result = await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: `Optimizing Triton kernel (${baselineClassName} → ${optimizedClassName})`,
        cancellable: true
    }, async (progress, token): Promise<GenerationResult> => {
        progress.report({ message: 'Analyzing baseline kernel...' });
        
        // Check for cancellation
        token.onCancellationRequested(() => {
            if (webviewProvider) {
                webviewProvider.updateHistory({
                    id: taskId,
                    timestamp: new Date(),
                    backend: 'triton',
                    status: 'cancelled',
                    fileName: `${fileName} (optimize)`
                });
                webviewProvider.setCurrentTask(null);
            }
        });
        
        try {
            // For triton2triton, we use the FULL document as the baseline code
            // This includes the user's Triton kernel + get_inputs/get_init_inputs
            // The baseline class (user's kernel) is used for accuracy and performance comparison
            
            // Call backend to optimize
            // Note: User's Triton kernel is the baseline, NOT PyTorch reference
            return await backend.optimize({
                code: fullDocumentCode,  // Full document contains baseline kernel + get_inputs
                maxAttempts: config.get<number>('maxAttempts') || 3,
                currentClassName: baselineClassName,   // User's kernel class
                newClassName: optimizedClassName       // New optimized class
            }, progress, (attempt, maxAttempts) => {
                // Update sidebar with current attempt
                if (webviewProvider) {
                    webviewProvider.updateHistory({
                        id: taskId,
                        timestamp: new Date(),
                        backend: 'triton',
                        status: 'generating',
                        fileName: `${fileName} (optimize)`,
                        attempt: attempt,
                        maxAttempts: maxAttempts
                    });
                }
            });
        } catch (error: any) {
            return { success: false, error: error.message };
        }
    });
    
    // Clear current task
    if (webviewProvider) {
        webviewProvider.setCurrentTask(null);
    }

    // Handle result OUTSIDE of withProgress to avoid blocking issues
    // Debug log - show important info to user
    console.log(`[OptimizeTriton] success=${result.success} code=${!!result.code} speedup=${result.evaluation?.speedup} accuracy=${result.evaluation?.accuracy_pass} target=${config.get<number>('targetSpeedup') || 1.0}`);
    
    // Update sidebar history with final result
    const speedupForOptHistory = result.evaluation?.speedup || 0;
    const optHistoryEntry: GenerationHistoryEntry = {
        id: taskId,
        timestamp: new Date(),
        backend: 'triton',
        status: result.success ? 'completed' : 'failed',
        fileName: `${fileName} (optimize)`,
        evaluation: result.evaluation ? {
            compiled: result.evaluation.compile_success || false,
            correctness: result.evaluation.accuracy_pass || false,
            speedup: speedupForOptHistory,
            runtime: result.evaluation.optimized_time_ms || result.evaluation.new_time_ms,
            refRuntime: result.evaluation.baseline_time_ms || result.evaluation.ref_time_ms
        } : undefined,
        error: result.error,
        code: result.code
    };
    if (webviewProvider) {
        webviewProvider.updateHistory(optHistoryEntry);
    }
    
    if (result.success && result.code) {
        // Check if speedup >= target (optimization actually improved performance)
        const speedup = result.evaluation?.speedup || 0;
        const accuracyPass = result.evaluation?.accuracy_pass ?? false;
        const targetSpeedup = config.get<number>('targetSpeedup') || 1.0;
        
        // More debug  
        console.log(`[OptimizeTriton] Decision: speedup=${speedup} >= ${targetSpeedup}? ${speedup >= targetSpeedup}, accuracy=${accuracyPass}, autoInsert=${speedup >= targetSpeedup && accuracyPass}`);
        
        const insertCode = async (warning: boolean) => {
            console.log(`[OptimizeTriton] INSERTING CODE warning=${warning}`);
            const lastLine = editor.document.lineCount - 1;
            const lastChar = editor.document.lineAt(lastLine).text.length;
            const endPosition = new vscode.Position(lastLine, lastChar);
            
            const separator = '\n\n# ' + '='.repeat(60) + '\n';
            const header = `# Optimized Triton Kernel: ${optimizedClassName} (${new Date().toLocaleString()})\n`;
            const classInfo = `# ${baselineClassName} → ${optimizedClassName}\n`;
            const evalComment = warning
                ? `# ⚠ WARNING: Speedup=${speedup.toFixed(2)}x (below target ${targetSpeedup}x)\n`
                : `# Evaluation: Compile=✓ Accuracy=✓ Speedup=${speedup.toFixed(2)}x\n`;
            
            const codeToInsert = separator + header + classInfo + evalComment + '# ' + '='.repeat(60) + '\n\n' + result.code;
            
            await editor.edit(editBuilder => {
                editBuilder.insert(endPosition, codeToInsert);
            });
            
            // Scroll to the inserted code
            const newLastLine = editor.document.lineCount - 1;
            editor.revealRange(
                new vscode.Range(endPosition, new vscode.Position(newLastLine, 0)),
                vscode.TextEditorRevealType.InCenter
            );
        };
        
        // Decision logic: only auto-insert if speedup >= target AND accuracy passes
        const shouldAutoInsert = speedup >= targetSpeedup && accuracyPass;
        
        if (shouldAutoInsert) {
            // SUCCESS: Insert optimized code
            await insertCode(false);
            vscode.window.showInformationMessage(
                `✓ Optimization successful! Speedup: ${speedup.toFixed(2)}x`
            );
        } else {
            // NO IMPROVEMENT: MUST ask user with modal dialog
            const baselineTime = result.evaluation?.baseline_time_ms || result.evaluation?.ref_time_ms || 0;
            const optimizedTime = result.evaluation?.optimized_time_ms || result.evaluation?.new_time_ms || 0;
            
            const action = await vscode.window.showWarningMessage(
                `⚠️ Performance: ${speedup.toFixed(2)}x (target: ${targetSpeedup}x), Accuracy: ${accuracyPass ? 'PASS' : 'FAIL'}`,
                { modal: true },  // Force modal dialog - user MUST respond
                'Insert Code',
                'Cancel'
            );
            
            if (action === 'Insert Code') {
                await insertCode(true);
            }
        }
        
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
}

/**
 * Extract the relevant generated code (ModelNew class and dependencies)
 * 
 * Strategy: Find the first Triton-related line and include everything from there.
 * This avoids complex parsing that can break multi-line function definitions.
 */
function extractRelevantCode(generatedCode: string): string {
    const lines = generatedCode.split('\n');
    
    // Find the start of Triton-related code
    let startIndex = -1;
    for (let i = 0; i < lines.length; i++) {
        const trimmed = lines[i].trim();
        
        // Look for the first Triton-related line
        if (trimmed.startsWith('import triton') ||
            trimmed.startsWith('from triton') ||
            trimmed.startsWith('@triton') ||
            (trimmed.startsWith('import os') && i + 1 < lines.length && lines[i + 1].includes('TRITON')) ||
            (trimmed.startsWith('os.environ') && (trimmed.includes('TRITON') || trimmed.includes('PYTORCH'))) ||
            trimmed.match(/^[A-Z_]+\s*=\s*\d+/) // Constants like NUM_XCDS
        ) {
            startIndex = i;
            break;
        }
    }
    
    // If no Triton code found, look for ModelNew class
    if (startIndex === -1) {
        for (let i = 0; i < lines.length; i++) {
            if (lines[i].trim().startsWith('class ModelNew')) {
                startIndex = i;
                break;
            }
        }
    }
    
    // If still nothing found, return the whole code
    if (startIndex === -1) {
        return generatedCode;
    }
    
    // Find the end: after ModelNew class or end of file
    let endIndex = lines.length;
    let inModelNew = false;
    let modelNewIndent = 0;
    
    for (let i = startIndex; i < lines.length; i++) {
        const line = lines[i];
        const trimmed = line.trim();
        
        if (trimmed.startsWith('class ModelNew')) {
            inModelNew = true;
            // Get the indentation of the class definition
            modelNewIndent = line.length - line.trimStart().length;
        }
        
        // Skip get_inputs and get_init_inputs functions (they're from the original problem)
        if (trimmed.startsWith('def get_inputs') || trimmed.startsWith('def get_init_inputs')) {
            endIndex = i;
            break;
        }
        
        // If we were in ModelNew and hit another top-level class/def (not indented), stop there
        if (inModelNew && i > startIndex && trimmed !== '') {
            const currentIndent = line.length - line.trimStart().length;
            if (currentIndent <= modelNewIndent && 
                !trimmed.startsWith('class ModelNew') &&
                (trimmed.startsWith('class ') || trimmed.startsWith('def ')) &&
                !trimmed.startsWith('def __init__') && !trimmed.startsWith('def forward')) {
                endIndex = i;
                break;
            }
        }
    }
    
    // Extract the relevant portion
    const relevantLines = lines.slice(startIndex, endIndex);
    
    // Clean up: remove trailing empty lines
    while (relevantLines.length > 0 && relevantLines[relevantLines.length - 1].trim() === '') {
        relevantLines.pop();
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

// Store generation history for sidebar panel
interface GenerationHistoryEntry {
    id: string;
    timestamp: Date;
    backend: 'triton' | 'hip';
    status: 'pending' | 'generating' | 'completed' | 'failed' | 'cancelled';
    fileName?: string;
    evaluation?: {
        compiled: boolean;
        correctness: boolean;
        speedup: number;
        runtime?: number;
        refRuntime?: number;
    };
    error?: string;
    code?: string;
    attempt?: number;
    maxAttempts?: number;
}

class GeneratorWebviewProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;
    private _history: GenerationHistoryEntry[] = [];
    private _currentTaskId: string | null = null;
    private _cancelRequested: boolean = false;

    constructor(
        private readonly _extensionUri: vscode.Uri,
        private readonly _backend: PythonBackend,
        private readonly _analyzer: CodeAnalyzer
    ) {}

    // Public method to add/update history from external commands
    public updateHistory(entry: GenerationHistoryEntry) {
        const existingIndex = this._history.findIndex(h => h.id === entry.id);
        if (existingIndex >= 0) {
            this._history[existingIndex] = entry;
        } else {
            this._history.unshift(entry); // Add to beginning
        }
        // Keep only last 20 entries
        if (this._history.length > 20) {
            this._history = this._history.slice(0, 20);
        }
        this._sendHistoryUpdate();
    }

    public setCurrentTask(taskId: string | null) {
        this._currentTaskId = taskId;
        this._cancelRequested = false;
        this._sendStatusUpdate();
    }

    public isCancelRequested(): boolean {
        return this._cancelRequested;
    }

    private _sendHistoryUpdate() {
        if (this._view) {
            this._view.webview.postMessage({
                command: 'historyUpdate',
                history: this._history
            });
        }
    }

    private _sendStatusUpdate() {
        if (this._view) {
            this._view.webview.postMessage({
                command: 'taskStatus',
                currentTaskId: this._currentTaskId,
                cancelRequested: this._cancelRequested
            });
        }
    }

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
                case 'openSettings':
                    vscode.commands.executeCommand('workbench.action.openSettings', 'hipGenerator');
                    break;
                case 'cancelTask':
                    this._cancelRequested = true;
                    vscode.window.showInformationMessage('Cancellation requested. Task will stop after current attempt.');
                    this._sendStatusUpdate();
                    break;
                case 'clearHistory':
                    this._history = [];
                    this._sendHistoryUpdate();
                    break;
                case 'viewCode':
                    const entry = this._history.find(h => h.id === message.id);
                    if (entry?.code) {
                        const doc = await vscode.workspace.openTextDocument({
                            content: entry.code,
                            language: 'python'
                        });
                        vscode.window.showTextDocument(doc, { preview: true });
                    }
                    break;
                case 'requestHistory':
                    this._sendHistoryUpdate();
                    this._sendStatusUpdate();
                    break;
            }
        });

        // Send initial state
        setTimeout(() => {
            this._sendHistoryUpdate();
            this._sendStatusUpdate();
        }, 100);
    }

    private _getHtmlForWebview(webview: vscode.Webview): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HIP Generator</title>
    <style>
        * { box-sizing: border-box; }
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
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .section-title button {
            width: auto;
            padding: 2px 8px;
            font-size: 11px;
            margin: 0;
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
        button.cancel-btn {
            background-color: var(--vscode-inputValidation-errorBackground);
            border: 1px solid var(--vscode-errorForeground);
            color: var(--vscode-errorForeground);
        }
        button.cancel-btn:hover {
            background-color: var(--vscode-errorForeground);
            color: var(--vscode-button-foreground);
        }
        .info {
            font-size: 12px;
            color: var(--vscode-descriptionForeground);
            margin-top: 8px;
        }
        .current-task {
            background-color: var(--vscode-inputValidation-infoBackground);
            border: 1px solid var(--vscode-inputValidation-infoBorder);
            border-radius: 4px;
            padding: 12px;
            margin-bottom: 16px;
        }
        .current-task h4 {
            margin: 0 0 8px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .spinner {
            width: 14px;
            height: 14px;
            border: 2px solid var(--vscode-foreground);
            border-top-color: transparent;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            display: inline-block;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .history-item {
            background-color: var(--vscode-editor-background);
            border: 1px solid var(--vscode-panel-border);
            border-radius: 4px;
            padding: 10px;
            margin-bottom: 8px;
        }
        .history-item:hover {
            border-color: var(--vscode-focusBorder);
        }
        .history-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        .history-header .time {
            font-size: 11px;
            color: var(--vscode-descriptionForeground);
        }
        .history-header .backend {
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 10px;
            font-weight: 500;
        }
        .backend-triton {
            background-color: var(--vscode-badge-background);
            color: var(--vscode-badge-foreground);
        }
        .backend-hip {
            background-color: var(--vscode-badge-background);
            color: var(--vscode-badge-foreground);
        }
        .history-status {
            display: flex;
            gap: 12px;
            font-size: 12px;
            margin-bottom: 6px;
        }
        .status-item {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .status-icon { font-size: 14px; }
        .status-icon.success { color: var(--vscode-charts-green, #4caf50); }
        .status-icon.error { color: var(--vscode-errorForeground, #f44336); }
        .status-icon.warning { color: var(--vscode-editorWarning-foreground, #ff9800); }
        .speedup {
            font-weight: bold;
            font-size: 14px;
        }
        .speedup.good { color: var(--vscode-charts-green, #4caf50); }
        .speedup.bad { color: var(--vscode-errorForeground, #f44336); }
        .speedup.neutral { color: var(--vscode-editorWarning-foreground, #ff9800); }
        .history-actions {
            display: flex;
            gap: 8px;
            margin-top: 8px;
        }
        .history-actions button {
            width: auto;
            padding: 4px 8px;
            font-size: 11px;
            margin: 0;
        }
        .file-name {
            font-size: 11px;
            color: var(--vscode-descriptionForeground);
            margin-bottom: 4px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .error-msg {
            font-size: 11px;
            color: var(--vscode-errorForeground);
            margin-top: 4px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .runtime-details {
            font-size: 11px;
            color: var(--vscode-descriptionForeground);
            margin-top: 4px;
        }
        .empty-state {
            text-align: center;
            padding: 20px;
            color: var(--vscode-descriptionForeground);
        }
        .empty-state p {
            margin: 8px 0;
        }
    </style>
</head>
<body>
    <div id="currentTask" class="current-task" style="display: none;">
        <h4><span class="spinner"></span> Generating...</h4>
        <p id="taskInfo" style="margin: 0; font-size: 12px;"></p>
        <button class="cancel-btn" onclick="cancelTask()" style="margin-top: 8px;">
            ⏹ Cancel Task
        </button>
    </div>

    <div class="section">
        <div class="section-title">⚙️ Settings</div>
        <button onclick="openSettings()">
            Configure API Key & Options
        </button>
    </div>
    
    <div class="info" style="margin-bottom: 16px;">
        <p><strong>Usage:</strong></p>
        <ol style="padding-left: 20px; margin: 8px 0;">
            <li>Select PyTorch code in editor</li>
            <li>Right-click → "Generate Triton/HIP Kernel"</li>
            <li>Or right-click → "Optimize Triton Kernel"</li>
        </ol>
    </div>

    <div class="section">
        <div class="section-title">
            📊 Generation History
            <button onclick="clearHistory()">Clear</button>
        </div>
        <div id="historyList">
            <div class="empty-state">
                <p>No generations yet</p>
                <p style="font-size: 11px;">Select code and right-click to generate</p>
            </div>
        </div>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        let history = [];
        let currentTaskId = null;
        
        // Request initial state
        vscode.postMessage({ command: 'requestHistory' });
        
        function openSettings() {
            vscode.postMessage({ command: 'openSettings' });
        }
        
        function cancelTask() {
            vscode.postMessage({ command: 'cancelTask' });
        }
        
        function clearHistory() {
            vscode.postMessage({ command: 'clearHistory' });
        }
        
        function viewCode(id) {
            vscode.postMessage({ command: 'viewCode', id: id });
        }
        
        function formatTime(timestamp) {
            const date = new Date(timestamp);
            const now = new Date();
            const diff = now - date;
            
            if (diff < 60000) return 'Just now';
            if (diff < 3600000) return Math.floor(diff / 60000) + 'm ago';
            if (diff < 86400000) return Math.floor(diff / 3600000) + 'h ago';
            return date.toLocaleDateString();
        }
        
        function renderHistory() {
            const container = document.getElementById('historyList');
            
            if (history.length === 0) {
                container.innerHTML = \`
                    <div class="empty-state">
                        <p>No generations yet</p>
                        <p style="font-size: 11px;">Select code and right-click to generate</p>
                    </div>
                \`;
                return;
            }
            
            container.innerHTML = history.map(item => {
                const backendClass = item.backend === 'triton' ? 'backend-triton' : 'backend-hip';
                const backendLabel = item.backend.toUpperCase();
                
                let statusHtml = '';
                if (item.status === 'generating') {
                    const attemptInfo = item.attempt && item.maxAttempts 
                        ? \` (Attempt \${item.attempt}/\${item.maxAttempts})\`
                        : '';
                    statusHtml = \`<span class="status-item"><span class="spinner" style="width:12px;height:12px;"></span> Generating\${attemptInfo}</span>\`;
                } else if (item.status === 'failed' || item.status === 'cancelled') {
                    statusHtml = \`<span class="status-item"><span class="status-icon error">✗</span> \${item.status === 'cancelled' ? 'Cancelled' : 'Failed'}</span>\`;
                } else if (item.evaluation) {
                    const e = item.evaluation;
                    const compileIcon = e.compiled ? '✓' : '✗';
                    const compileClass = e.compiled ? 'success' : 'error';
                    const correctIcon = e.correctness ? '✓' : '✗';
                    const correctClass = e.correctness ? 'success' : 'error';
                    
                    let speedupHtml = '';
                    if (e.speedup !== undefined && e.speedup > 0) {
                        const speedupClass = e.speedup >= 1.0 ? 'good' : (e.speedup >= 0.5 ? 'neutral' : 'bad');
                        const speedupIcon = e.speedup >= 1.0 ? '🚀' : '⚠️';
                        speedupHtml = \`<span class="status-item"><span class="speedup \${speedupClass}">\${speedupIcon} \${e.speedup.toFixed(2)}x</span></span>\`;
                    }
                    
                    statusHtml = \`
                        <span class="status-item"><span class="status-icon \${compileClass}">\${compileIcon}</span> Compile</span>
                        <span class="status-item"><span class="status-icon \${correctClass}">\${correctIcon}</span> Correct</span>
                        \${speedupHtml}
                    \`;
                }
                
                const fileName = item.fileName ? \`<div class="file-name">📄 \${item.fileName}</div>\` : '';
                const errorMsg = item.error ? \`<div class="error-msg">❌ \${item.error.substring(0, 80)}...</div>\` : '';
                
                let runtimeDetails = '';
                if (item.evaluation && item.evaluation.runtime && item.evaluation.refRuntime) {
                    runtimeDetails = \`<div class="runtime-details">⏱ Ref: \${item.evaluation.refRuntime.toFixed(2)}ms → Gen: \${item.evaluation.runtime.toFixed(2)}ms</div>\`;
                }
                
                const viewBtn = item.code ? \`<button onclick="viewCode('\${item.id}')">👁 View Code</button>\` : '';
                
                return \`
                    <div class="history-item">
                        <div class="history-header">
                            <span class="backend \${backendClass}">\${backendLabel}</span>
                            <span class="time">\${formatTime(item.timestamp)}</span>
                        </div>
                        \${fileName}
                        <div class="history-status">\${statusHtml}</div>
                        \${runtimeDetails}
                        \${errorMsg}
                        \${viewBtn ? \`<div class="history-actions">\${viewBtn}</div>\` : ''}
                    </div>
                \`;
            }).join('');
        }
        
        function updateCurrentTask() {
            const taskEl = document.getElementById('currentTask');
            const taskInfo = document.getElementById('taskInfo');
            
            if (currentTaskId) {
                const current = history.find(h => h.id === currentTaskId);
                if (current && current.status === 'generating') {
                    taskEl.style.display = 'block';
                    const attemptInfo = current.attempt && current.maxAttempts 
                        ? \`Attempt \${current.attempt}/\${current.maxAttempts}\`
                        : 'Processing...';
                    taskInfo.textContent = \`\${current.backend.toUpperCase()} kernel - \${attemptInfo}\`;
                    return;
                }
            }
            taskEl.style.display = 'none';
        }

        window.addEventListener('message', event => {
            const message = event.data;
            
            switch (message.command) {
                case 'historyUpdate':
                    history = message.history || [];
                    renderHistory();
                    updateCurrentTask();
                    break;
                case 'taskStatus':
                    currentTaskId = message.currentTaskId;
                    updateCurrentTask();
                    break;
            }
        });
    </script>
</body>
</html>`;
    }
}

// Global reference to webview provider for external access
let webviewProvider: GeneratorWebviewProvider;

export function deactivate() {
    if (backend) {
        backend.dispose();
    }
}
