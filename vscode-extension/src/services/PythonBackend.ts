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
    baseline_time_ms?: number;  // For triton2triton
    optimized_time_ms?: number;  // For triton2triton
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
        progress: vscode.Progress<{ message?: string; increment?: number }>,
        attemptCallback?: (attempt: number, maxAttempts: number) => void
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

        // Track state for retry logic
        let prevCodeFile: string | undefined;
        let prevEvalResult: EvaluationResult | undefined;
        let retryMode: 'reflection' | 'optimization' | undefined;

        try {
            for (let attempt = 1; attempt <= maxAttempts; attempt++) {
                progress.report({ 
                    message: `Attempt ${attempt}/${maxAttempts}: Calling LLM...`,
                    increment: 0
                });
                
                // Notify caller of current attempt
                if (attemptCallback) {
                    attemptCallback(attempt, maxAttempts);
                }

                this._outputChannel.appendLine(`\n--- Attempt ${attempt}/${maxAttempts} ---`);

                const outputFile = path.join(outputDir, `code_${attempt}.py`);
                let generateResult: { success: boolean; error?: string };

                // First attempt: normal generation; subsequent: use retry mode
                if (attempt === 1) {
                    // Initial generation
                    generateResult = await this._runGenerate(
                        pythonPath,
                        hipGeneratorPath,
                        problemFile,
                        outputFile,
                        request.backend,
                        apiKey,
                        config.get<number>('temperature') || 0.3
                    );
                } else if (prevCodeFile && retryMode) {
                    // Retry with reflection or optimization
                    this._outputChannel.appendLine(`Using ${retryMode} mode for retry`);
                    
                    generateResult = await this._runRetry(
                        pythonPath,
                        hipGeneratorPath,
                        problemFile,
                        prevCodeFile,
                        outputFile,
                        request.backend,
                        apiKey,
                        config.get<number>('temperature') || 0.3,
                        retryMode,
                        attempt,
                        {
                            errorMsg: lastError,
                            speedup: prevEvalResult?.speedup,
                            refCodeFile: problemFile  // Use problem file as reference
                        }
                    );
                } else {
                    // Fallback to normal generation
                    generateResult = await this._runGenerate(
                        pythonPath,
                        hipGeneratorPath,
                        problemFile,
                        outputFile,
                        request.backend,
                        apiKey,
                        config.get<number>('temperature') || 0.3
                    );
                }

                if (!generateResult.success || !fs.existsSync(outputFile)) {
                    lastError = generateResult.error || 'Generation failed';
                    this._outputChannel.appendLine(`Generation failed: ${lastError}`);
                    progress.report({ 
                        message: `Attempt ${attempt}/${maxAttempts}: Generation failed, retrying...`,
                        increment: 100 / maxAttempts / 2
                    });
                    // Use reflection mode for next attempt
                    retryMode = 'reflection';
                    continue;
                }

                const generatedCode = fs.readFileSync(outputFile, 'utf-8');
                this._outputChannel.appendLine(`Generated ${generatedCode.length} chars`);
                prevCodeFile = outputFile;

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
                prevEvalResult = evalResult;

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

                // Decide retry mode based on eval result
                if (!evalResult.compile_success) {
                    // Compilation failed - use reflection
                    retryMode = 'reflection';
                    lastError = evalResult.error || 'Compilation failed';
                    this._outputChannel.appendLine(`  → Next retry: reflection (compile error)`);
                } else if (!evalResult.accuracy_pass) {
                    // Accuracy failed - use reflection
                    retryMode = 'reflection';
                    lastError = evalResult.error || 
                        (evalResult.has_nan ? 'Output contains NaN' : 
                         evalResult.has_inf ? 'Output contains Inf' : 
                         `Max diff too high: ${evalResult.max_diff}`);
                    this._outputChannel.appendLine(`  → Next retry: reflection (accuracy error)`);
                } else if (evalResult.speedup < (config.get<number>('targetSpeedup') || 1.0)) {
                    // Correct but slow - use optimization
                    retryMode = 'optimization';
                    const target = config.get<number>('targetSpeedup') || 1.0;
                    this._outputChannel.appendLine(`  → Next retry: optimization (speedup=${evalResult.speedup.toFixed(2)}x < target ${target}x)`);
                }

                // Track best result
                if (evalResult.accuracy_pass) {
                    if (!bestEval || evalResult.speedup > bestEval.speedup) {
                        bestCode = generatedCode;
                        bestEval = evalResult;
                    }
                    
                    // If we got a good result (speedup >= target), we can stop
                    const targetSpeedup = config.get<number>('targetSpeedup') || 1.0;
                    if (evalResult.speedup >= targetSpeedup) {
                        this._outputChannel.appendLine(`✓ Target achieved! Speedup: ${evalResult.speedup.toFixed(2)}x (target: ${targetSpeedup}x)`);
                        break;
                    }
                } else if (!bestCode) {
                    // Keep the code even if accuracy failed (user might want to debug)
                    bestCode = generatedCode;
                    bestEval = evalResult;
                }
            }

            // Return result
            if (bestCode) {
                const result = {
                    success: bestEval?.accuracy_pass || false,
                    code: bestCode,
                    evaluation: bestEval,
                    attempts: maxAttempts,
                    error: bestEval?.accuracy_pass ? undefined : lastError
                };
                this._outputChannel.appendLine(`\n=== FINAL RESULT ===`);
                this._outputChannel.appendLine(`success=${result.success}`);
                this._outputChannel.appendLine(`speedup=${result.evaluation?.speedup}`);
                this._outputChannel.appendLine(`accuracy_pass=${result.evaluation?.accuracy_pass}`);
                this._outputChannel.appendLine(`===================\n`);
                return result;
            } else {
                this._outputChannel.appendLine(`\n=== FINAL RESULT: NO CODE ===\n`);
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
     * Optimize existing Triton kernel code (triton2triton)
     * 
     * Key principle: User's Triton kernel is the BASELINE
     * - Accuracy: compare optimized output vs user's kernel output
     * - Performance: compare optimized time vs user's kernel time
     * 
     * Flow: profile baseline → generate optimized → eval (vs baseline) → repeat
     */
    async optimize(
        request: OptimizeRequest,
        progress: vscode.Progress<{ message?: string; increment?: number }>,
        attemptCallback?: (attempt: number, maxAttempts: number) => void
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

        // Class name handling for triton2triton
        // User's kernel class is the BASELINE
        const baselineClassName = request.currentClassName || 'ModelNew';
        // Generate a unique name for the optimized kernel (must NOT conflict with baseline)
        const optimizedClassName = request.newClassName || baselineClassName + 'Optimized';
        
        // Ensure names don't conflict
        if (optimizedClassName === baselineClassName) {
            return {
                success: false,
                error: `Optimized class name '${optimizedClassName}' cannot be the same as baseline '${baselineClassName}'`
            };
        }

        // Create baseline code file (user's Triton kernel is the baseline)
        const baselineCodeFile = path.join(this._tempDir, `baseline_${Date.now()}.py`);
        fs.writeFileSync(baselineCodeFile, request.code);
        
        const outputDir = path.join(this._tempDir, `optimize_output_${Date.now()}`);
        fs.mkdirSync(outputDir, { recursive: true });

        this._outputChannel.appendLine(`\n${'='.repeat(60)}`);
        this._outputChannel.appendLine(`Triton2Triton Optimization`);
        this._outputChannel.appendLine(`Baseline class: ${baselineClassName} (user's kernel)`);
        this._outputChannel.appendLine(`Optimized class: ${optimizedClassName} (to be generated)`);
        this._outputChannel.appendLine(`Max attempts: ${maxAttempts}`);
        this._outputChannel.appendLine(`Note: Baseline IS the user's Triton kernel (not PyTorch reference)`);
        this._outputChannel.show(true);

        let bestCode: string | undefined;
        let bestEval: EvaluationResult | undefined;
        let lastError: string | undefined;
        let feedback: string = request.feedback || '';

        try {
            // Step 1: Profile the baseline (user's kernel)
            progress.report({ 
                message: `Step 1: Profiling baseline kernel (${baselineClassName})...`,
                increment: 0
            });

            this._outputChannel.appendLine(`\n--- Step 1: Profiling Baseline Kernel ---`);
            
            // Run baseline kernel to get performance metrics
            const baselineEval = await this._evaluateTritonBaseline(
                baselineCodeFile,
                baselineClassName,
                pythonPath,
                hipGeneratorPath
            );
            
            if (baselineEval.compile_success) {
                this._outputChannel.appendLine(
                    `Baseline performance: ${baselineEval.baseline_time_ms?.toFixed(3) || 'N/A'} ms`
                );
                
                if (baselineEval.profiler_feedback) {
                    feedback = baselineEval.profiler_feedback;
                    this._outputChannel.appendLine(`Profiler feedback obtained`);
                } else {
                    // Build feedback for optimization
                    feedback = this._buildTritonOptimizeFeedback(baselineEval);
                }
                
                progress.report({ 
                    message: `Baseline: ${baselineEval.baseline_time_ms?.toFixed(3) || 'N/A'} ms`,
                    increment: 10
                });
            } else {
                this._outputChannel.appendLine(`⚠️ Baseline kernel has issues: ${baselineEval.error?.substring(0, 200) || 'unknown'}`);
                this._outputChannel.appendLine(`Proceeding with optimization anyway...`);
                feedback = `The current kernel has issues: ${baselineEval.error || 'compilation error'}. Please fix these issues while optimizing.`;
            }

            // Step 2+: Generate → Eval loop (ALWAYS run all attempts, keep best)
            // Key change: Don't break early even if speedup >= 1.0
            // Generate multiple attempts and pick the best one
            const allResults: Array<{code: string; eval: EvaluationResult & { profiler_feedback?: string }}> = [];
            
            for (let attempt = 1; attempt <= maxAttempts; attempt++) {
                progress.report({ 
                    message: `Attempt ${attempt}/${maxAttempts}: Generating optimized kernel...`,
                    increment: 0
                });
                
                // Notify caller of current attempt
                if (attemptCallback) {
                    attemptCallback(attempt, maxAttempts);
                }

                this._outputChannel.appendLine(`\n--- Optimization Attempt ${attempt}/${maxAttempts} ---`);
                if (feedback) {
                    this._outputChannel.appendLine(`Using feedback (${feedback.length} chars)`);
                }

                // Generate optimized code
                const outputFile = path.join(outputDir, `optimized_${attempt}.py`);
                const generateResult = await this._runOptimize(
                    pythonPath,
                    hipGeneratorPath,
                    baselineCodeFile,
                    outputFile,
                    apiKey,
                    config.get<number>('temperature') || 0.3,
                    feedback,
                    optimizedClassName  // New class name for optimized kernel
                );

                if (!generateResult.success || !fs.existsSync(outputFile)) {
                    lastError = generateResult.error || 'Optimization failed';
                    this._outputChannel.appendLine(`Generation failed: ${lastError}`);
                    progress.report({ 
                        message: `Attempt ${attempt}/${maxAttempts}: LLM call failed, continuing...`,
                        increment: 100 / maxAttempts / 3
                    });
                    feedback = `Previous attempt failed with error: ${lastError}. Please fix and try again.`;
                    continue;
                }

                const optimizedCode = fs.readFileSync(outputFile, 'utf-8');
                this._outputChannel.appendLine(`Generated optimized code: ${optimizedCode.length} chars`);

                // Basic validation
                if (optimizedCode.length < 100 || 
                    (!optimizedCode.includes('@triton') && !optimizedCode.includes('triton.jit'))) {
                    lastError = 'Generated code does not appear to be valid Triton code';
                    this._outputChannel.appendLine(`Warning: ${lastError}`);
                    feedback = `Previous output was not valid Triton code. Please generate proper Triton kernel code with @triton.jit decorator.`;
                    progress.report({ 
                        message: `Attempt ${attempt}/${maxAttempts}: Invalid code, continuing...`,
                        increment: 100 / maxAttempts / 3
                    });
                    continue;
                }

                // Verify the optimized code contains the correct class name
                if (!optimizedCode.includes(`class ${optimizedClassName}`)) {
                    this._outputChannel.appendLine(`Warning: Generated code missing class ${optimizedClassName}`);
                    feedback = `Generated code must contain class ${optimizedClassName}. Do not reuse the baseline class name ${baselineClassName}.`;
                    continue;
                }

                // Evaluate: compare optimized kernel vs baseline kernel
                progress.report({ 
                    message: `Attempt ${attempt}/${maxAttempts}: Evaluating vs baseline...`,
                    increment: 100 / maxAttempts / 3
                });

                const evalResult = await this._evaluateTritonOptimize(
                    baselineCodeFile,       // User's kernel as baseline
                    outputFile,             // Generated optimized kernel
                    baselineClassName,      // User's class name
                    optimizedClassName,     // New optimized class name
                    pythonPath,
                    hipGeneratorPath
                );

                const statusEmoji = evalResult.compile_success 
                    ? (evalResult.accuracy_pass ? '✓' : '⚠') 
                    : '✗';
                
                this._outputChannel.appendLine(
                    `  Compile: ${evalResult.compile_success ? '✓' : '✗'} | ` +
                    `Accuracy: ${evalResult.accuracy_pass ? '✓' : '✗'} | ` +
                    `Speedup vs baseline: ${evalResult.speedup?.toFixed(2) || 'N/A'}x`
                );

                progress.report({ 
                    message: `Attempt ${attempt}/${maxAttempts}: ${statusEmoji} ` +
                        `Compile=${evalResult.compile_success ? '✓' : '✗'} ` +
                        `Accuracy=${evalResult.accuracy_pass ? '✓' : '✗'} ` +
                        `Speedup=${evalResult.speedup?.toFixed(2) || 'N/A'}x`,
                    increment: 100 / maxAttempts / 3
                });

                // Store all results
                allResults.push({ code: optimizedCode, eval: evalResult });

                // Get target speedup from config
                const targetSpeedup = config.get<number>('targetSpeedup') || 1.0;

                // Build feedback based on result for next attempt
                if (evalResult.compile_success && evalResult.accuracy_pass) {
                    // Success - but continue to find better
                    if ((evalResult.speedup || 0) >= targetSpeedup) {
                        this._outputChannel.appendLine(`  Good result! Speedup: ${evalResult.speedup?.toFixed(2)}x >= target ${targetSpeedup}x (continuing to find better)`);
                        feedback = `Previous attempt achieved ${evalResult.speedup?.toFixed(2)}x speedup. Try to achieve even better performance with more aggressive optimizations.`;
                    } else {
                        feedback = evalResult.profiler_feedback || 
                            `The optimized kernel speedup (${evalResult.speedup?.toFixed(2)}x) is below target (${targetSpeedup}x). Please optimize further.`;
                    }
                } else {
                    // Build feedback based on failure type
                    if (!evalResult.compile_success) {
                        feedback = `Compilation error: ${evalResult.error || 'unknown'}. Please fix the syntax and try again.`;
                        lastError = evalResult.error || 'Compilation failed';
                    } else {
                        feedback = `Accuracy check failed (max_diff=${evalResult.max_diff?.toFixed(6)}). ` +
                            `The optimized kernel must produce the same output as the baseline kernel.`;
                        lastError = evalResult.error || 
                            (evalResult.has_nan ? 'Output contains NaN' : 
                             evalResult.has_inf ? 'Output contains Inf' : 
                             `Max diff too high: ${evalResult.max_diff}`);
                    }
                }
            }
            
            // Select best result from all attempts
            // Filter: accuracy must pass, then pick highest speedup
            const validResults = allResults.filter(r => r.eval.compile_success && r.eval.accuracy_pass);
            
            this._outputChannel.appendLine(`\n--- Results Summary ---`);
            this._outputChannel.appendLine(`Total attempts: ${maxAttempts}`);
            this._outputChannel.appendLine(`Valid results (accuracy pass): ${validResults.length}`);
            
            if (validResults.length > 0) {
                // Sort by speedup descending
                validResults.sort((a, b) => (b.eval.speedup || 0) - (a.eval.speedup || 0));
                
                // Log all valid results
                validResults.forEach((r, i) => {
                    this._outputChannel.appendLine(`  ${i + 1}. Speedup: ${r.eval.speedup?.toFixed(2)}x`);
                });
                
                // Pick the best one
                bestCode = validResults[0].code;
                bestEval = validResults[0].eval;
                this._outputChannel.appendLine(`\n✓ Selected best result: ${bestEval.speedup?.toFixed(2)}x speedup`);
            } else if (allResults.length > 0) {
                // No valid results, keep the last one for debugging
                bestCode = allResults[allResults.length - 1].code;
                bestEval = allResults[allResults.length - 1].eval;
                this._outputChannel.appendLine(`⚠ No valid results, keeping last attempt for debugging`);
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
            this._cleanup(baselineCodeFile);
            try {
                fs.rmSync(outputDir, { recursive: true, force: true });
            } catch {}
        }
    }
    
    /**
     * Build feedback for triton2triton optimization
     */
    private _buildTritonOptimizeFeedback(evalResult: any): string {
        const parts: string[] = [];
        parts.push('=== BASELINE KERNEL ANALYSIS ===');
        
        if (evalResult.baseline_time_ms) {
            parts.push(`Baseline execution time: ${evalResult.baseline_time_ms.toFixed(3)} ms`);
        }
        
        parts.push('\n=== OPTIMIZATION GOALS ===');
        parts.push('1. Maintain numerical accuracy (output must match baseline)');
        parts.push('2. Improve execution time compared to baseline');
        
        parts.push('\n=== MI350 OPTIMIZATION TECHNIQUES ===');
        parts.push('- XCD Swizzle (32 XCDs): Remap block IDs for better distribution');
        parts.push('- L2 Cache Grouping (GROUP_M=8): Group tiles for cache locality');
        parts.push('- MFMA: Use matrix_instr_nonkdim=16 for 16x16 instructions');
        parts.push('- Block sizes: Try 128x128x64 or 256x128x64 for GEMM');
        parts.push('- Environment: TRITON_HIP_USE_BLOCK_PINGPONG=1, TRITON_HIP_USE_ASYNC_COPY=1');
        
        return parts.join('\n');
    }
    
    /**
     * Profile baseline Triton kernel (run and get performance)
     */
    private async _evaluateTritonBaseline(
        codeFile: string,
        baselineClass: string,
        pythonPath: string,
        hipGeneratorPath: string
    ): Promise<EvaluationResult & { baseline_time_ms?: number; profiler_feedback?: string }> {
        const resultFile = path.join(this._tempDir, `baseline_result_${Date.now()}.json`);
        
        this._outputChannel.appendLine(`Profiling baseline kernel: ${baselineClass}`);
        
        try {
            const evalScript = path.join(hipGeneratorPath, 'eval.py');
            // Use triton-optimize mode with same file as both baseline and optimized
            // This just runs the baseline kernel to get timing
            const args = [
                evalScript,
                '--triton-optimize',
                '--code', codeFile,
                '--problem', codeFile,  // Same file (contains get_inputs)
                '--baseline-code', codeFile,
                '--baseline-class', baselineClass,
                '--optimized-class', baselineClass,  // Same class (just timing)
                '--output', resultFile,
                '--profile'
            ];

            const config = vscode.workspace.getConfiguration('hipGenerator');
            const apiKey = config.get<string>('amdApiKey');

            await this._runPython(pythonPath, args, {
                LLM_GATEWAY_KEY: apiKey || '',
                PYTORCH_ROCM_ARCH: 'gfx950'
            });

            if (fs.existsSync(resultFile)) {
                let jsonContent = fs.readFileSync(resultFile, 'utf-8');
                jsonContent = this._fixJsonSpecialValues(jsonContent);
                const evalResult = JSON.parse(jsonContent);
                return {
                    ...evalResult,
                    baseline_time_ms: evalResult.baseline_time_ms || evalResult.ref_time_ms,
                    profiler_feedback: evalResult.profiler_feedback
                };
            }
            
            return {
                ...this._failedEvaluation('Baseline profiling failed'),
                baseline_time_ms: undefined
            };
        } catch (error: any) {
            return {
                ...this._failedEvaluation(error.message),
                baseline_time_ms: undefined
            };
        } finally {
            this._cleanup(resultFile);
        }
    }
    
    /**
     * Evaluate optimized kernel vs baseline (triton2triton)
     */
    private async _evaluateTritonOptimize(
        baselineCodeFile: string,
        optimizedCodeFile: string,
        baselineClass: string,
        optimizedClass: string,
        pythonPath: string,
        hipGeneratorPath: string
    ): Promise<EvaluationResult & { profiler_feedback?: string }> {
        const resultFile = path.join(this._tempDir, `triton_opt_result_${Date.now()}.json`);
        
        this._outputChannel.appendLine(`Evaluating: ${optimizedClass} vs baseline ${baselineClass}`);
        
        try {
            const evalScript = path.join(hipGeneratorPath, 'eval.py');
            const args = [
                evalScript,
                '--triton-optimize',
                '--code', optimizedCodeFile,
                '--problem', baselineCodeFile,  // Baseline file has get_inputs
                '--baseline-code', baselineCodeFile,
                '--baseline-class', baselineClass,
                '--optimized-class', optimizedClass,
                '--output', resultFile,
                '--profile'
            ];

            const config = vscode.workspace.getConfiguration('hipGenerator');
            const apiKey = config.get<string>('amdApiKey');

            await this._runPython(pythonPath, args, {
                LLM_GATEWAY_KEY: apiKey || '',
                PYTORCH_ROCM_ARCH: 'gfx950'
            });

            if (fs.existsSync(resultFile)) {
                let jsonContent = fs.readFileSync(resultFile, 'utf-8');
                jsonContent = this._fixJsonSpecialValues(jsonContent);
                const evalResult = JSON.parse(jsonContent);
                return {
                    ...evalResult,
                    profiler_feedback: evalResult.profiler_feedback
                };
            }
            
            return {
                ...this._failedEvaluation('Evaluation failed - no result file'),
                profiler_feedback: undefined
            };
        } catch (error: any) {
            return {
                ...this._failedEvaluation(error.message),
                profiler_feedback: undefined
            };
        } finally {
            this._cleanup(resultFile);
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

    /**
     * Run generate.py in retry mode (reflection or optimization)
     */
    private async _runRetry(
        pythonPath: string,
        hipGeneratorPath: string,
        problemFile: string,
        prevCodeFile: string,
        outputFile: string,
        backend: string,
        apiKey: string,
        temperature: number,
        retryMode: 'reflection' | 'optimization',
        attempt: number,
        options: {
            errorMsg?: string;  // For reflection mode
            speedup?: number;   // For optimization mode
            refCodeFile?: string;  // For optimization mode
        }
    ): Promise<{ success: boolean; error?: string }> {
        const generateScript = path.join(hipGeneratorPath, 'generate.py');
        const args = [
            generateScript,
            '--problem', problemFile,
            '--prev-code', prevCodeFile,
            '--output', outputFile,
            '--backend', backend,
            '--temperature', temperature.toString(),
            '--retry-mode', retryMode,
            '--attempt', attempt.toString()
        ];

        if (retryMode === 'reflection' && options.errorMsg) {
            args.push('--error-msg', options.errorMsg);
        } else if (retryMode === 'optimization') {
            if (options.speedup !== undefined) {
                args.push('--speedup', options.speedup.toString());
            }
            if (options.refCodeFile) {
                args.push('--ref-code', options.refCodeFile);
            }
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
        // Check paths in order of preference:
        // 1. Bundled within extension (for packaged vsix)
        // 2. Parent directory (for development)
        // 3. Common installation paths
        const possiblePaths = [
            // Bundled in extension's hipgenerator folder
            path.join(this._context.extensionPath, 'hipgenerator'),
            // Parent directory (development mode)
            path.join(this._context.extensionPath, '..'),
            path.join(__dirname, '..', '..', '..'),
            // Fallback to common paths
            '/root/HipGenerator',
            process.cwd()
        ];

        for (const p of possiblePaths) {
            const generatePy = path.join(p, 'generate.py');
            if (fs.existsSync(generatePy)) {
                this._outputChannel.appendLine(`Found HipGenerator at: ${p}`);
                return p;
            }
        }
        
        this._outputChannel.appendLine('ERROR: HipGenerator not found. Searched paths:');
        possiblePaths.forEach(p => this._outputChannel.appendLine(`  - ${p}`));

        return null;
    }

    private async _runPython(
        pythonPath: string,
        args: string[],
        env: Record<string, string>,
        timeoutMs: number = 300000  // 5 min default timeout
    ): Promise<{ exitCode: number; stdout: string; stderr: string }> {
        return new Promise((resolve) => {
            const proc = cp.spawn(pythonPath, args, {
                env: { ...process.env, ...env },
                cwd: this._getHipGeneratorPath() || undefined
            });

            let stdout = '';
            let stderr = '';
            let killed = false;

            // Set timeout
            const timeout = setTimeout(() => {
                killed = true;
                proc.kill('SIGKILL');
                this._outputChannel.appendLine(`\n⚠ Process killed after ${timeoutMs / 1000}s timeout`);
            }, timeoutMs);

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
                clearTimeout(timeout);
                resolve({
                    exitCode: killed ? 124 : (code || 0),  // 124 = timeout
                    stdout,
                    stderr: killed ? 'Process killed due to timeout' : stderr
                });
            });

            proc.on('error', (error) => {
                clearTimeout(timeout);
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
