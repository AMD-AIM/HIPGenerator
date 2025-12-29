import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';

export interface GenerationRequest {
    code: string;
    backend: 'triton' | 'hip';
    getInputs?: string;
    getInitInputs?: string;
    maxAttempts?: number;
}

export interface OptimizeRequest {
    code: string;              // Triton code to optimize
    originalCode?: string;     // Original PyTorch code (for evaluation reference)
    getInputs?: string;        // get_inputs() function
    getInitInputs?: string;    // get_init_inputs() function
    maxAttempts?: number;
    feedback?: string;         // Feedback from previous optimization attempt
    currentClassName?: string; // Current wrapper class name (e.g., ModelNew)
    newClassName?: string;     // New class name for optimized code (e.g., ModelNewNew)
    refClassName?: string;     // Reference class name for evaluation (e.g., Model)
}

export interface GenerationResult {
    success: boolean;
    code?: string;
    error?: string;
    evaluation?: EvaluationResult;
    attempts?: number;
}

export interface EvaluationRequest {
    generatedCode: string;
    originalCode: string;
    getInputs: string;
    getInitInputs: string;
    backend: 'triton' | 'hip';
}

export interface EvaluationResult {
    compile_success: boolean;
    accuracy_pass: boolean;
    max_diff: number;
    mean_diff: number;
    has_nan: boolean;
    has_inf: boolean;
    ref_time_ms: number;
    new_time_ms: number;
    speedup: number;
    error?: string;
}

export type ProgressCallback = (message: string, increment?: number) => void;

export class PythonBackend {
    private _context: vscode.ExtensionContext;
    private _outputChannel: vscode.OutputChannel;
    private _tempDir: string;

    constructor(context: vscode.ExtensionContext) {
        this._context = context;
        this._outputChannel = vscode.window.createOutputChannel('HIP Generator');
        this._tempDir = path.join(os.tmpdir(), 'hip-generator');
        
        // Ensure temp directory exists
        if (!fs.existsSync(this._tempDir)) {
            fs.mkdirSync(this._tempDir, { recursive: true });
        }
    }

    /**
     * Generate kernel code with progress reporting
     */
    async generateWithProgress(
        request: GenerationRequest,
        progress: vscode.Progress<{ message?: string; increment?: number }>
    ): Promise<GenerationResult> {
        const config = vscode.workspace.getConfiguration('hipGenerator');
        const apiKey = config.get<string>('amdApiKey');
        const pythonPath = config.get<string>('pythonPath') || 'python3';
        const maxAttempts = request.maxAttempts || config.get<number>('maxAttempts') || 3;
        
        if (!apiKey) {
            return {
                success: false,
                error: 'AMD API Key not configured. Please set hipGenerator.amdApiKey in settings.'
            };
        }

        const hipGeneratorPath = this._getHipGeneratorPath();
        if (!hipGeneratorPath) {
            return {
                success: false,
                error: 'HipGenerator not found. Please ensure the extension is installed correctly.'
            };
        }

        // Create problem file
        const problemFile = await this._createProblemFile(request);
        const outputDir = path.join(this._tempDir, `output_${Date.now()}`);
        fs.mkdirSync(outputDir, { recursive: true });

        this._outputChannel.appendLine(`\n${'='.repeat(60)}`);
        this._outputChannel.appendLine(`Generating ${request.backend.toUpperCase()} kernel...`);
        this._outputChannel.appendLine(`Max attempts: ${maxAttempts}`);
        this._outputChannel.show(true);

        let bestCode: string | undefined;
        let bestEval: EvaluationResult | undefined;
        let lastError: string | undefined;

        try {
            for (let attempt = 1; attempt <= maxAttempts; attempt++) {
                progress.report({ 
                    message: `Attempt ${attempt}/${maxAttempts}: Calling LLM...`,
                    increment: 0
                });

                this._outputChannel.appendLine(`\n--- Attempt ${attempt}/${maxAttempts} ---`);

                // Generate code
                const outputFile = path.join(outputDir, `code_${attempt}.py`);
                const generateResult = await this._runGenerate(
                    pythonPath,
                    hipGeneratorPath,
                    problemFile,
                    outputFile,
                    request.backend,
                    apiKey,
                    config.get<number>('temperature') || 0.3
                );

                if (!generateResult.success || !fs.existsSync(outputFile)) {
                    lastError = generateResult.error || 'Generation failed';
                    this._outputChannel.appendLine(`Generation failed: ${lastError}`);
                    progress.report({ 
                        message: `Attempt ${attempt}/${maxAttempts}: Generation failed, retrying...`,
                        increment: 100 / maxAttempts / 2
                    });
                    continue;
                }

                const generatedCode = fs.readFileSync(outputFile, 'utf-8');
                this._outputChannel.appendLine(`Generated ${generatedCode.length} chars`);

                // Evaluate
                progress.report({ 
                    message: `Attempt ${attempt}/${maxAttempts}: Evaluating...`,
                    increment: 100 / maxAttempts / 4
                });

                const evalResult = await this._evaluate(
                    generatedCode,
                    problemFile,
                    request.backend,
                    pythonPath,
                    hipGeneratorPath
                );

                const statusEmoji = evalResult.compile_success 
                    ? (evalResult.accuracy_pass ? '✓' : '⚠') 
                    : '✗';
                
                this._outputChannel.appendLine(
                    `  Compile: ${evalResult.compile_success ? '✓' : '✗'} | ` +
                    `Accuracy: ${evalResult.accuracy_pass ? '✓' : '✗'} | ` +
                    `Speedup: ${evalResult.speedup?.toFixed(2) || 'N/A'}x`
                );

                progress.report({ 
                    message: `Attempt ${attempt}/${maxAttempts}: ${statusEmoji} ` +
                        `Compile=${evalResult.compile_success ? '✓' : '✗'} ` +
                        `Accuracy=${evalResult.accuracy_pass ? '✓' : '✗'} ` +
                        `Speedup=${evalResult.speedup?.toFixed(2) || 'N/A'}x`,
                    increment: 100 / maxAttempts / 4
                });

                // Track best result
                if (evalResult.accuracy_pass) {
                    if (!bestEval || evalResult.speedup > bestEval.speedup) {
                        bestCode = generatedCode;
                        bestEval = evalResult;
                    }
                    
                    // If we got a good result (speedup >= 1.0), we can stop
                    if (evalResult.speedup >= 1.0) {
                        this._outputChannel.appendLine(`✓ Target achieved! Speedup: ${evalResult.speedup.toFixed(2)}x`);
                        break;
                    }
                } else if (!bestCode) {
                    // Keep the code even if accuracy failed (user might want to debug)
                    bestCode = generatedCode;
                    bestEval = evalResult;
                    lastError = evalResult.error || 
                        (evalResult.has_nan ? 'Output contains NaN' : 
                         evalResult.has_inf ? 'Output contains Inf' : 
                         `Max diff too high: ${evalResult.max_diff}`);
                }
            }

            // Return result
            if (bestCode) {
                return {
                    success: bestEval?.accuracy_pass || false,
                    code: bestCode,
                    evaluation: bestEval,
                    attempts: maxAttempts,
                    error: bestEval?.accuracy_pass ? undefined : lastError
                };
            } else {
                return {
                    success: false,
                    error: lastError || 'All attempts failed'
                };
            }

        } catch (error: any) {
            this._outputChannel.appendLine(`Error: ${error.message}`);
            return {
                success: false,
                error: error.message
            };
        } finally {
            // Cleanup
            this._cleanup(problemFile);
            try {
                fs.rmSync(outputDir, { recursive: true, force: true });
            } catch {}
        }
    }

    /**
     * Simple generate without progress (backwards compatible)
     */
    async generate(request: GenerationRequest): Promise<GenerationResult> {
        // Wrap in a dummy progress
        return this.generateWithProgress(request, {
            report: () => {}
        });
    }

    /**
     * Optimize existing Triton kernel code with evaluation loop
     * Flow: eval (get profiler feedback) → generate → eval → repeat
     */
    async optimize(
        request: OptimizeRequest,
        progress: vscode.Progress<{ message?: string; increment?: number }>
    ): Promise<GenerationResult> {
        const config = vscode.workspace.getConfiguration('hipGenerator');
        const apiKey = config.get<string>('amdApiKey');
        const pythonPath = config.get<string>('pythonPath') || 'python3';
        const maxAttempts = request.maxAttempts || config.get<number>('maxAttempts') || 3;
        
        if (!apiKey) {
            return {
                success: false,
                error: 'AMD API Key not configured. Please set hipGenerator.amdApiKey in settings.'
            };
        }

        const hipGeneratorPath = this._getHipGeneratorPath();
        if (!hipGeneratorPath) {
            return {
                success: false,
                error: 'HipGenerator not found. Please ensure the extension is installed correctly.'
            };
        }

        // Class name handling
        const currentClassName = request.currentClassName || 'ModelNew';
        const newClassName = request.newClassName || currentClassName + 'New';
        const refClassName = request.refClassName || 'Model';

        // Create input code file with existing Triton code
        const inputCodeFile = path.join(this._tempDir, `optimize_input_${Date.now()}.py`);
        fs.writeFileSync(inputCodeFile, request.code);
        
        // Create problem file for evaluation (original PyTorch code + get_inputs)
        let problemFile: string | undefined;
        if (request.originalCode) {
            problemFile = path.join(this._tempDir, `optimize_problem_${Date.now()}.py`);
            let problemContent = request.originalCode;
            if (request.getInputs && !problemContent.includes('def get_inputs()')) {
                problemContent += '\n\n' + request.getInputs;
            }
            if (request.getInitInputs && !problemContent.includes('def get_init_inputs()')) {
                problemContent += '\n\n' + request.getInitInputs;
            }
            fs.writeFileSync(problemFile, problemContent);
        }
        
        const outputDir = path.join(this._tempDir, `optimize_output_${Date.now()}`);
        fs.mkdirSync(outputDir, { recursive: true });

        this._outputChannel.appendLine(`\n${'='.repeat(60)}`);
        this._outputChannel.appendLine(`Optimizing Triton kernel (triton2triton)...`);
        this._outputChannel.appendLine(`Current class: ${currentClassName} → New class: ${newClassName}`);
        this._outputChannel.appendLine(`Reference class: ${refClassName}`);
        this._outputChannel.appendLine(`Max attempts: ${maxAttempts}`);
        this._outputChannel.appendLine(`Has reference for evaluation: ${problemFile ? 'Yes' : 'No'}`);
        this._outputChannel.show(true);

        let bestCode: string | undefined;
        let bestEval: EvaluationResult | undefined;
        let lastError: string | undefined;
        let feedback: string = request.feedback || '';

        try {
            // Step 1: Initial evaluation of current code (get profiler feedback)
            if (problemFile) {
                progress.report({ 
                    message: `Step 1: Evaluating current kernel with profiler...`,
                    increment: 0
                });

                this._outputChannel.appendLine(`\n--- Step 1: Initial Profiler Analysis ---`);
                
                // For initial eval, we need to use the full input code file (not just selected code)
                // because it needs to contain both the kernel and the ModelNew class
                const initialEval = await this._evaluateWithProfiler(
                    fs.readFileSync(inputCodeFile, 'utf-8'),  // Use full input file
                    problemFile,
                    'triton',
                    pythonPath,
                    hipGeneratorPath,
                    refClassName,
                    currentClassName
                );

                if (initialEval.compile_success) {
                    this._outputChannel.appendLine(
                        `Current performance: Speedup=${initialEval.speedup?.toFixed(2) || 'N/A'}x`
                    );
                    
                    // Use profiler feedback if available
                    if (initialEval.profiler_feedback) {
                        feedback = initialEval.profiler_feedback;
                        this._outputChannel.appendLine(`Profiler feedback obtained`);
                    } else {
                        // Build basic feedback
                        feedback = this._buildOptimizeFeedback(initialEval, 
                            initialEval.accuracy_pass ? 'performance' : 'accuracy');
                    }
                    
                    progress.report({ 
                        message: `Current: Speedup=${initialEval.speedup?.toFixed(2) || 'N/A'}x`,
                        increment: 10
                    });
                } else {
                    this._outputChannel.appendLine(`Current code has compile error: ${initialEval.error?.substring(0, 200) || 'unknown'}`);
                    feedback = this._buildOptimizeFeedback(initialEval, 'compile');
                }
            } else {
                this._outputChannel.appendLine(`⚠️ No reference Model class found - skipping initial evaluation`);
                this._outputChannel.appendLine(`For best results, include a reference 'class Model(nn.Module)' in your file`);
                progress.report({ 
                    message: `No reference found, proceeding with optimization...`,
                    increment: 10
                });
            }

            // Step 2+: Generate → Eval loop
            for (let attempt = 1; attempt <= maxAttempts; attempt++) {
                progress.report({ 
                    message: `Attempt ${attempt}/${maxAttempts}: Calling LLM for optimization...`,
                    increment: 0
                });

                this._outputChannel.appendLine(`\n--- Optimization Attempt ${attempt}/${maxAttempts} ---`);
                if (feedback) {
                    this._outputChannel.appendLine(`Using feedback (${feedback.length} chars)`);
                }

                // Generate optimized code
                const outputFile = path.join(outputDir, `optimized_${attempt}.py`);
                const generateResult = await this._runOptimize(
                    pythonPath,
                    hipGeneratorPath,
                    inputCodeFile,
                    outputFile,
                    apiKey,
                    config.get<number>('temperature') || 0.3,
                    feedback,
                    newClassName
                );

                if (!generateResult.success || !fs.existsSync(outputFile)) {
                    lastError = generateResult.error || 'Optimization failed';
                    this._outputChannel.appendLine(`Optimization failed: ${lastError}`);
                    progress.report({ 
                        message: `Attempt ${attempt}/${maxAttempts}: LLM call failed, retrying...`,
                        increment: 100 / maxAttempts / 3
                    });
                    feedback = `Previous attempt failed with error: ${lastError}. Please fix and try again.`;
                    continue;
                }

                const optimizedCode = fs.readFileSync(outputFile, 'utf-8');
                this._outputChannel.appendLine(`Generated optimized code: ${optimizedCode.length} chars`);

                // Basic validation - check if we got valid Python/Triton code
                if (optimizedCode.length < 100 || 
                    (!optimizedCode.includes('@triton') && !optimizedCode.includes('triton.jit'))) {
                    lastError = 'Generated code does not appear to be valid Triton code';
                    this._outputChannel.appendLine(`Warning: ${lastError}`);
                    feedback = `Previous output was not valid Triton code. Please generate proper Triton kernel code with @triton.jit decorator.`;
                    progress.report({ 
                        message: `Attempt ${attempt}/${maxAttempts}: Invalid code, retrying...`,
                        increment: 100 / maxAttempts / 3
                    });
                    continue;
                }

                // Evaluate with profiler if we have a reference
                if (problemFile) {
                    progress.report({ 
                        message: `Attempt ${attempt}/${maxAttempts}: Evaluating with profiler...`,
                        increment: 100 / maxAttempts / 3
                    });

                    const evalResult = await this._evaluateWithProfiler(
                        optimizedCode,
                        problemFile,
                        'triton',
                        pythonPath,
                        hipGeneratorPath,
                        refClassName,
                        newClassName
                    );

                    const statusEmoji = evalResult.compile_success 
                        ? (evalResult.accuracy_pass ? '✓' : '⚠') 
                        : '✗';
                    
                    this._outputChannel.appendLine(
                        `  Compile: ${evalResult.compile_success ? '✓' : '✗'} | ` +
                        `Accuracy: ${evalResult.accuracy_pass ? '✓' : '✗'} | ` +
                        `Speedup: ${evalResult.speedup?.toFixed(2) || 'N/A'}x`
                    );

                    progress.report({ 
                        message: `Attempt ${attempt}/${maxAttempts}: ${statusEmoji} ` +
                            `Compile=${evalResult.compile_success ? '✓' : '✗'} ` +
                            `Accuracy=${evalResult.accuracy_pass ? '✓' : '✗'} ` +
                            `Speedup=${evalResult.speedup?.toFixed(2) || 'N/A'}x`,
                        increment: 100 / maxAttempts / 3
                    });

                    // Track best result
                    if (evalResult.compile_success && evalResult.accuracy_pass) {
                        if (!bestEval || evalResult.speedup > bestEval.speedup) {
                            bestCode = optimizedCode;
                            bestEval = evalResult;
                        }
                        
                        // If we got a good speedup, we can stop
                        if (evalResult.speedup >= 1.0) {
                            this._outputChannel.appendLine(`✓ Target achieved! Speedup: ${evalResult.speedup.toFixed(2)}x`);
                            break;
                        } else {
                            // Use profiler feedback for next attempt
                            feedback = evalResult.profiler_feedback || 
                                this._buildOptimizeFeedback(evalResult, 'performance');
                        }
                    } else {
                        // Build feedback based on failure type
                        if (!evalResult.compile_success) {
                            feedback = this._buildOptimizeFeedback(evalResult, 'compile');
                            lastError = evalResult.error || 'Compilation failed';
                        } else {
                            feedback = evalResult.profiler_feedback || 
                                this._buildOptimizeFeedback(evalResult, 'accuracy');
                            lastError = evalResult.error || 
                                (evalResult.has_nan ? 'Output contains NaN' : 
                                 evalResult.has_inf ? 'Output contains Inf' : 
                                 `Max diff too high: ${evalResult.max_diff}`);
                        }
                        
                        // Keep the code even if it failed (user might want to debug)
                        if (!bestCode) {
                            bestCode = optimizedCode;
                            bestEval = evalResult;
                        }
                    }
                } else {
                    // No evaluation reference - just accept the optimized code
                    bestCode = optimizedCode;
                    this._outputChannel.appendLine(`✓ Optimization complete (no evaluation reference)`);
                    
                    progress.report({ 
                        message: `Attempt ${attempt}/${maxAttempts}: ✓ Optimization complete`,
                        increment: 100 / maxAttempts
                    });
                    break;
                }
            }

            // Return result
            if (bestCode) {
                return {
                    success: bestEval?.accuracy_pass ?? true,
                    code: bestCode,
                    evaluation: bestEval,
                    attempts: maxAttempts,
                    error: (bestEval?.accuracy_pass ?? true) ? undefined : lastError
                };
            } else {
                return {
                    success: false,
                    error: lastError || 'All optimization attempts failed'
                };
            }

        } catch (error: any) {
            this._outputChannel.appendLine(`Error: ${error.message}`);
            return {
                success: false,
                error: error.message
            };
        } finally {
            // Cleanup
            this._cleanup(inputCodeFile);
            if (problemFile) {
                this._cleanup(problemFile);
            }
            try {
                fs.rmSync(outputDir, { recursive: true, force: true });
            } catch {}
        }
    }

    /**
     * Build feedback message for optimization based on evaluation result
     */
    private _buildOptimizeFeedback(evalResult: EvaluationResult, failureType: 'compile' | 'accuracy' | 'performance'): string {
        let feedback = '';
        
        switch (failureType) {
            case 'compile':
                feedback = `COMPILATION FAILED!\nError: ${evalResult.error || 'Unknown compile error'}\n\n`;
                feedback += `Please fix the compilation error and ensure:\n`;
                feedback += `1. All imports are correct (import triton, import triton.language as tl)\n`;
                feedback += `2. Use tl.minimum() instead of Python's min() inside @triton.jit functions\n`;
                feedback += `3. All tl.constexpr parameters are properly typed\n`;
                feedback += `4. No Python built-in functions inside JIT kernels`;
                break;
                
            case 'accuracy':
                feedback = `ACCURACY TEST FAILED!\n`;
                if (evalResult.has_nan) {
                    feedback += `Output contains NaN values!\n\n`;
                    feedback += `Common causes:\n`;
                    feedback += `1. Division by zero in kernel\n`;
                    feedback += `2. Invalid pointer arithmetic\n`;
                    feedback += `3. Out-of-bounds memory access\n`;
                } else if (evalResult.has_inf) {
                    feedback += `Output contains Infinity values!\n\n`;
                    feedback += `Common causes:\n`;
                    feedback += `1. Overflow in computation\n`;
                    feedback += `2. Missing boundary checks\n`;
                } else {
                    feedback += `Max diff: ${evalResult.max_diff} (too high)\n\n`;
                    feedback += `Common causes:\n`;
                    feedback += `1. Wrong stride calculation for transposed matrices\n`;
                    feedback += `2. Incorrect accumulator initialization\n`;
                    feedback += `3. Wrong output dtype conversion\n`;
                }
                feedback += `\nPlease fix the accuracy issue while preserving optimizations.`;
                break;
                
            case 'performance':
                feedback = `PERFORMANCE NOT OPTIMAL\n`;
                feedback += `Current speedup: ${evalResult.speedup?.toFixed(2)}x (target: >= 1.0x)\n\n`;
                feedback += `Try these MI350 optimizations:\n`;
                feedback += `1. Add XCD Swizzle (NUM_XCDS = 32 for MI350)\n`;
                feedback += `2. Add L2 Cache Grouping (GROUP_M parameter)\n`;
                feedback += `3. Use matrix_instr_nonkdim=16 for 16x16 MFMA\n`;
                feedback += `4. Set TRITON_HIP_USE_BLOCK_PINGPONG='1'\n`;
                feedback += `5. Precompute grid and strides in __init__\n`;
                feedback += `6. Try larger block sizes (256x256x32 for large GEMM)`;
                break;
        }
        
        return feedback;
    }

    private async _runOptimize(
        pythonPath: string,
        hipGeneratorPath: string,
        codeFile: string,
        outputFile: string,
        apiKey: string,
        temperature: number,
        feedback?: string,
        newClassName?: string
    ): Promise<{ success: boolean; error?: string }> {
        const generateScript = path.join(hipGeneratorPath, 'generate.py');
        const args = [
            generateScript,
            '--optimize', codeFile,
            '--output', outputFile,
            '--backend', 'triton',
            '--num-samples', '1',
            '--temperature', temperature.toString()
        ];

        // Add feedback if available
        if (feedback) {
            args.push('--feedback', feedback);
        }

        // Add new class name if specified
        if (newClassName) {
            args.push('--new-class-name', newClassName);
        }

        const result = await this._runPython(pythonPath, args, {
            LLM_GATEWAY_KEY: apiKey,
            PYTORCH_ROCM_ARCH: 'gfx950'
        });

        return {
            success: result.exitCode === 0,
            error: result.stderr || undefined
        };
    }

    /**
     * Evaluate code with profiler to get detailed feedback
     */
    private async _evaluateWithProfiler(
        generatedCode: string,
        problemFile: string,
        backend: string,
        pythonPath: string,
        hipGeneratorPath: string,
        refClassName: string = 'Model',
        newClassName: string = 'ModelNew'
    ): Promise<EvaluationResult & { profiler_feedback?: string }> {
        const codeFile = path.join(this._tempDir, `eval_code_${Date.now()}.py`);
        const resultFile = path.join(this._tempDir, `result_${Date.now()}.json`);
        
        fs.writeFileSync(codeFile, generatedCode);

        this._outputChannel.appendLine(`Evaluating with profiler (ref=${refClassName}, new=${newClassName})...`);

        try {
            const evalScript = path.join(hipGeneratorPath, 'eval.py');
            const args = [
                evalScript,
                '--code', codeFile,
                '--problem', problemFile,
                '--output', resultFile,
                '--backend', backend,
                '--profile',  // Enable profiler
                '--ref-class', refClassName,
                '--new-class', newClassName
            ];

            const config = vscode.workspace.getConfiguration('hipGenerator');
            const apiKey = config.get<string>('amdApiKey');

            const result = await this._runPython(pythonPath, args, {
                LLM_GATEWAY_KEY: apiKey || '',
                PYTORCH_ROCM_ARCH: 'gfx950'
            });

            if (fs.existsSync(resultFile)) {
                try {
                    // Read and fix JSON (handle Infinity, NaN, etc.)
                    let jsonContent = fs.readFileSync(resultFile, 'utf-8');
                    jsonContent = this._fixJsonSpecialValues(jsonContent);
                    
                    const evalResult = JSON.parse(jsonContent);
                    return {
                        ...evalResult,
                        profiler_feedback: evalResult.profiler_feedback || undefined
                    };
                } catch (parseError: any) {
                    this._outputChannel.appendLine(`JSON parse error: ${parseError.message}`);
                    return {
                        ...this._failedEvaluation(`JSON parse error: ${parseError.message}`),
                        profiler_feedback: undefined
                    };
                }
            } else {
                // Try to extract error from stderr
                const errorMatch = result.stderr?.match(/Error:?\s*(.+)/i);
                const error = errorMatch ? errorMatch[1] : (result.stderr?.slice(-500) || 'Evaluation failed');
                return {
                    ...this._failedEvaluation(error),
                    profiler_feedback: undefined
                };
            }
        } finally {
            this._cleanup(codeFile, resultFile);
        }
    }

    private async _runGenerate(
        pythonPath: string,
        hipGeneratorPath: string,
        problemFile: string,
        outputFile: string,
        backend: string,
        apiKey: string,
        temperature: number
    ): Promise<{ success: boolean; error?: string }> {
        const generateScript = path.join(hipGeneratorPath, 'generate.py');
        const args = [
            generateScript,
            '--problem', problemFile,
            '--output', outputFile,
            '--backend', backend,
            '--num-samples', '1',
            '--temperature', temperature.toString()
        ];

        const result = await this._runPython(pythonPath, args, {
            LLM_GATEWAY_KEY: apiKey,
            PYTORCH_ROCM_ARCH: 'gfx950'
        });

        return {
            success: result.exitCode === 0,
            error: result.stderr || undefined
        };
    }

    async evaluate(request: EvaluationRequest): Promise<EvaluationResult> {
        const config = vscode.workspace.getConfiguration('hipGenerator');
        const pythonPath = config.get<string>('pythonPath') || 'python3';
        
        const hipGeneratorPath = this._getHipGeneratorPath();
        if (!hipGeneratorPath) {
            return this._failedEvaluation('HipGenerator not found');
        }

        const problemFile = await this._createProblemFile({
            code: request.originalCode,
            backend: request.backend,
            getInputs: request.getInputs,
            getInitInputs: request.getInitInputs
        });
        
        const codeFile = path.join(this._tempDir, `code_${Date.now()}.py`);
        fs.writeFileSync(codeFile, request.generatedCode);

        try {
            return await this._evaluate(
                request.generatedCode,
                problemFile,
                request.backend,
                pythonPath,
                hipGeneratorPath
            );
        } finally {
            this._cleanup(problemFile, codeFile);
        }
    }

    private _failedEvaluation(error: string): EvaluationResult {
        return {
            compile_success: false,
            accuracy_pass: false,
            max_diff: Infinity,
            mean_diff: Infinity,
            has_nan: true,
            has_inf: true,
            ref_time_ms: 0,
            new_time_ms: 0,
            speedup: 0,
            error
        };
    }

    private async _evaluate(
        generatedCode: string,
        problemFile: string,
        backend: string,
        pythonPath: string,
        hipGeneratorPath: string
    ): Promise<EvaluationResult> {
        const codeFile = path.join(this._tempDir, `eval_code_${Date.now()}.py`);
        const resultFile = path.join(this._tempDir, `result_${Date.now()}.json`);
        
        fs.writeFileSync(codeFile, generatedCode);

        this._outputChannel.appendLine(`Evaluating...`);

        try {
            const evalScript = path.join(hipGeneratorPath, 'eval.py');
            const args = [
                evalScript,
                '--code', codeFile,
                '--problem', problemFile,
                '--output', resultFile,
                '--backend', backend
            ];

            const config = vscode.workspace.getConfiguration('hipGenerator');
            const apiKey = config.get<string>('amdApiKey');

            const result = await this._runPython(pythonPath, args, {
                LLM_GATEWAY_KEY: apiKey || '',
                PYTORCH_ROCM_ARCH: 'gfx950'
            });

            if (fs.existsSync(resultFile)) {
                try {
                    // Read and fix JSON (handle Infinity, NaN, etc.)
                    let jsonContent = fs.readFileSync(resultFile, 'utf-8');
                    jsonContent = this._fixJsonSpecialValues(jsonContent);
                    
                    const evalResult = JSON.parse(jsonContent);
                    return evalResult;
                } catch (parseError: any) {
                    this._outputChannel.appendLine(`JSON parse error: ${parseError.message}`);
                    return this._failedEvaluation(`JSON parse error: ${parseError.message}`);
                }
            } else {
                // Try to extract error from stderr
                const errorMatch = result.stderr?.match(/Error:?\s*(.+)/i);
                const error = errorMatch ? errorMatch[1] : (result.stderr?.slice(-500) || 'Evaluation failed');
                return this._failedEvaluation(error);
            }
        } finally {
            this._cleanup(codeFile, resultFile);
        }
    }

    /**
     * Fix special JSON values that Python's json.dump produces but JS can't parse
     */
    private _fixJsonSpecialValues(json: string): string {
        // Replace Python's Infinity/NaN with valid JSON
        return json
            .replace(/:\s*Infinity/g, ': 1e308')
            .replace(/:\s*-Infinity/g, ': -1e308')
            .replace(/:\s*NaN/g, ': null')
            .replace(/"max_diff":\s*Infinity/g, '"max_diff": 1e308')
            .replace(/"mean_diff":\s*Infinity/g, '"mean_diff": 1e308');
    }

    private async _createProblemFile(request: GenerationRequest): Promise<string> {
        const problemFile = path.join(this._tempDir, `problem_${Date.now()}.py`);
        
        let content = request.code;
        
        if (request.getInputs && !content.includes('def get_inputs()')) {
            content += '\n\n' + request.getInputs;
        }
        
        if (request.getInitInputs && !content.includes('def get_init_inputs()')) {
            content += '\n\n' + request.getInitInputs;
        }
        
        fs.writeFileSync(problemFile, content);
        return problemFile;
    }

    private _getHipGeneratorPath(): string | null {
        const possiblePaths = [
            path.join(this._context.extensionPath, '..'),
            path.join(__dirname, '..', '..', '..'),
            '/root/HipGenerator',
            process.cwd()
        ];

        for (const p of possiblePaths) {
            const generatePy = path.join(p, 'generate.py');
            if (fs.existsSync(generatePy)) {
                return p;
            }
        }

        return null;
    }

    private async _runPython(
        pythonPath: string,
        args: string[],
        env: Record<string, string>
    ): Promise<{ exitCode: number; stdout: string; stderr: string }> {
        return new Promise((resolve) => {
            const proc = cp.spawn(pythonPath, args, {
                env: { ...process.env, ...env },
                cwd: this._getHipGeneratorPath() || undefined
            });

            let stdout = '';
            let stderr = '';

            proc.stdout.on('data', (data) => {
                const text = data.toString();
                stdout += text;
                this._outputChannel.append(text);
            });

            proc.stderr.on('data', (data) => {
                const text = data.toString();
                stderr += text;
                this._outputChannel.append(text);
            });

            proc.on('close', (code) => {
                resolve({
                    exitCode: code || 0,
                    stdout,
                    stderr
                });
            });

            proc.on('error', (error) => {
                resolve({
                    exitCode: 1,
                    stdout,
                    stderr: error.message
                });
            });
        });
    }

    private _cleanup(...files: string[]) {
        for (const file of files) {
            try {
                if (fs.existsSync(file)) {
                    fs.unlinkSync(file);
                }
            } catch {
                // Ignore cleanup errors
            }
        }
    }

    dispose() {
        this._outputChannel.dispose();
        
        try {
            if (fs.existsSync(this._tempDir)) {
                fs.rmSync(this._tempDir, { recursive: true, force: true });
            }
        } catch {
            // Ignore
        }
    }
}
