/**
 * CodeAnalyzer - Analyzes PyTorch code to infer get_inputs() and get_init_inputs() functions.
 * 
 * This analyzer examines:
 * 1. Model.__init__ parameters to determine constructor arguments
 * 2. Model.forward() signature to determine input tensor shapes
 * 3. Type annotations and docstrings for dtype information
 * 4. Common patterns like nn.Linear, nn.Conv2d to infer shapes
 */

export interface CodeAnalysis {
    modelName: string;
    initParams: ParameterInfo[];
    forwardParams: ParameterInfo[];
    getInputs: string;
    getInitInputs: string;
    inferredShapes: ShapeInfo[];
    warnings: string[];
}

export interface ParameterInfo {
    name: string;
    type?: string;
    defaultValue?: string;
    inferredShape?: string;
    inferredDtype?: string;
}

export interface ShapeInfo {
    paramName: string;
    shape: number[];
    dtype: string;
    device: string;
}

export class CodeAnalyzer {
    
    analyzeCode(code: string): CodeAnalysis {
        const result: CodeAnalysis = {
            modelName: 'Model',
            initParams: [],
            forwardParams: [],
            getInputs: '',
            getInitInputs: '',
            inferredShapes: [],
            warnings: []
        };

        try {
            // Extract class name
            result.modelName = this._extractModelName(code);
            
            // Extract __init__ parameters
            result.initParams = this._extractInitParams(code);
            
            // Extract forward parameters
            result.forwardParams = this._extractForwardParams(code);
            
            // Infer shapes from the code context
            result.inferredShapes = this._inferShapes(code, result.forwardParams);
            
            // Generate get_inputs() function
            result.getInputs = this._generateGetInputs(result.forwardParams, result.inferredShapes, code);
            
            // Generate get_init_inputs() function
            result.getInitInputs = this._generateGetInitInputs(result.initParams, code);
            
        } catch (error: any) {
            result.warnings.push(`Analysis error: ${error.message}`);
        }

        return result;
    }

    private _extractModelName(code: string): string {
        // Match class definition that inherits from nn.Module
        const classMatch = code.match(/class\s+(\w+)\s*\(\s*(?:nn\.Module|torch\.nn\.Module)\s*\)/);
        return classMatch ? classMatch[1] : 'Model';
    }

    private _extractInitParams(code: string): ParameterInfo[] {
        const params: ParameterInfo[] = [];
        
        // Find __init__ method
        const initMatch = code.match(/def\s+__init__\s*\(\s*self\s*(?:,\s*([^)]*))?\s*\)/);
        if (!initMatch || !initMatch[1]) {
            return params;
        }

        const paramStr = initMatch[1];
        const paramParts = this._splitParams(paramStr);
        
        for (const part of paramParts) {
            const param = this._parseParameter(part);
            if (param) {
                params.push(param);
            }
        }

        return params;
    }

    private _extractForwardParams(code: string): ParameterInfo[] {
        const params: ParameterInfo[] = [];
        
        // Find forward method
        const forwardMatch = code.match(/def\s+forward\s*\(\s*self\s*(?:,\s*([^)]*))?\s*\)/);
        if (!forwardMatch || !forwardMatch[1]) {
            return params;
        }

        const paramStr = forwardMatch[1];
        const paramParts = this._splitParams(paramStr);
        
        for (const part of paramParts) {
            const param = this._parseParameter(part);
            if (param) {
                params.push(param);
            }
        }

        return params;
    }

    private _splitParams(paramStr: string): string[] {
        // Split parameters while respecting brackets
        const params: string[] = [];
        let current = '';
        let depth = 0;
        
        for (const char of paramStr) {
            if (char === '[' || char === '(' || char === '{') {
                depth++;
            } else if (char === ']' || char === ')' || char === '}') {
                depth--;
            } else if (char === ',' && depth === 0) {
                if (current.trim()) {
                    params.push(current.trim());
                }
                current = '';
                continue;
            }
            current += char;
        }
        
        if (current.trim()) {
            params.push(current.trim());
        }
        
        return params;
    }

    private _parseParameter(paramStr: string): ParameterInfo | null {
        // Parse parameter with optional type annotation and default value
        // Examples: "x", "x: torch.Tensor", "dim: int = 64"
        
        const parts = paramStr.split('=');
        const nameAndType = parts[0].trim();
        const defaultValue = parts.length > 1 ? parts[1].trim() : undefined;
        
        const typeMatch = nameAndType.match(/^(\w+)\s*:\s*(.+)$/);
        if (typeMatch) {
            return {
                name: typeMatch[1],
                type: typeMatch[2].trim(),
                defaultValue
            };
        }
        
        return {
            name: nameAndType,
            defaultValue
        };
    }

    private _inferShapes(code: string, forwardParams: ParameterInfo[]): ShapeInfo[] {
        const shapes: ShapeInfo[] = [];
        
        // Default dtype and device
        const defaultDtype = this._inferDefaultDtype(code);
        const defaultDevice = 'cuda';
        
        for (const param of forwardParams) {
            const shape = this._inferShapeForParam(param, code);
            shapes.push({
                paramName: param.name,
                shape: shape,
                dtype: defaultDtype,
                device: defaultDevice
            });
        }
        
        return shapes;
    }

    private _inferDefaultDtype(code: string): string {
        // Check for bfloat16 mentions
        if (code.includes('bfloat16') || code.includes('bf16')) {
            return 'torch.bfloat16';
        }
        // Check for float16
        if (code.includes('float16') || code.includes('half')) {
            return 'torch.float16';
        }
        // Default to bfloat16 for AMD GPUs
        return 'torch.bfloat16';
    }

    private _inferShapeForParam(param: ParameterInfo, code: string): number[] {
        // Try to infer shape from various patterns
        
        // Look for explicit shape definitions in comments or docstrings
        const shapeCommentMatch = code.match(new RegExp(`${param.name}.*?\\[([\\d,\\s]+)\\]`, 'i'));
        if (shapeCommentMatch) {
            const shape = shapeCommentMatch[1].split(',').map(s => parseInt(s.trim()));
            if (shape.every(n => !isNaN(n))) {
                return shape;
            }
        }
        
        // Look for torch.randn/torch.zeros patterns with this param name
        const randnMatch = code.match(/torch\.(?:randn|zeros|ones|empty)\s*\(\s*(\d+)\s*,\s*(\d+)/);
        if (randnMatch) {
            return [parseInt(randnMatch[1]), parseInt(randnMatch[2])];
        }
        
        // Look for nn.Linear to infer matrix dimensions
        const linearMatch = code.match(/nn\.Linear\s*\(\s*(\d+)\s*,\s*(\d+)/);
        if (linearMatch) {
            const inFeatures = parseInt(linearMatch[1]);
            // Common batch size assumption
            return [1024, inFeatures];
        }
        
        // Look for matmul patterns to infer dimensions
        if (code.includes('matmul') || code.includes('mm') || code.includes('@')) {
            // Default GEMM dimensions
            return [1024, 1024];
        }
        
        // Look for explicit dimension variables
        const dimMatch = code.match(/(?:M|N|K|dim|size)\s*=\s*(\d+)/g);
        if (dimMatch && dimMatch.length >= 2) {
            const dims = dimMatch.map(m => parseInt(m.split('=')[1].trim()));
            return dims.slice(0, 2);
        }
        
        // Default fallback
        return [1024, 1024];
    }

    private _generateGetInputs(
        params: ParameterInfo[], 
        shapes: ShapeInfo[],
        code: string
    ): string {
        if (params.length === 0) {
            return `def get_inputs():
    return []`;
        }

        // Check if get_inputs already exists in code
        if (code.includes('def get_inputs()')) {
            // Extract existing get_inputs
            const match = code.match(/def get_inputs\(\):\s*\n((?:.*\n)*?return\s+\[[\s\S]*?\])/);
            if (match) {
                return `def get_inputs():\n${match[1]}`;
            }
        }

        const inputLines: string[] = [];
        
        for (let i = 0; i < params.length; i++) {
            const param = params[i];
            const shape = shapes[i] || { shape: [1024, 1024], dtype: 'torch.bfloat16', device: 'cuda' };
            
            // Generate tensor creation
            const shapeStr = shape.shape.join(', ');
            inputLines.push(`    torch.randn(${shapeStr}, dtype=${shape.dtype}, device='${shape.device}')`);
        }

        return `def get_inputs():
    return [
${inputLines.join(',\n')}
    ]`;
    }

    private _generateGetInitInputs(params: ParameterInfo[], code: string): string {
        if (params.length === 0) {
            return `def get_init_inputs():
    return []`;
        }

        // Check if get_init_inputs already exists in code
        if (code.includes('def get_init_inputs()')) {
            const match = code.match(/def get_init_inputs\(\):\s*\n((?:.*\n)*?return\s+\[[\s\S]*?\])/);
            if (match) {
                return `def get_init_inputs():\n${match[1]}`;
            }
        }

        const initArgs: string[] = [];
        
        for (const param of params) {
            if (param.defaultValue) {
                // Use default value if provided
                initArgs.push(`    ${param.defaultValue}  # ${param.name}`);
            } else {
                // Try to infer reasonable defaults based on name/type
                const value = this._inferInitValue(param, code);
                initArgs.push(`    ${value}  # ${param.name}`);
            }
        }

        return `def get_init_inputs():
    return [
${initArgs.join(',\n')}
    ]`;
    }

    private _inferInitValue(param: ParameterInfo, code: string): string {
        const name = param.name.toLowerCase();
        const type = param.type?.toLowerCase() || '';
        
        // Common parameter patterns
        if (name.includes('dim') || name.includes('size') || name.includes('hidden')) {
            // Look for value in code
            const match = code.match(new RegExp(`${param.name}\\s*=\\s*(\\d+)`, 'i'));
            if (match) {
                return match[1];
            }
            return '1024';
        }
        
        if (name.includes('num') || name.includes('n_')) {
            const match = code.match(new RegExp(`${param.name}\\s*=\\s*(\\d+)`, 'i'));
            if (match) {
                return match[1];
            }
            return '8';
        }
        
        if (name.includes('dropout') || name.includes('rate')) {
            return '0.0';
        }
        
        if (type.includes('bool')) {
            return 'True';
        }
        
        if (type.includes('float')) {
            return '0.0';
        }
        
        if (type.includes('int')) {
            return '1024';
        }
        
        if (type.includes('str')) {
            return '""';
        }
        
        // Check if used with nn.Linear or similar
        const linearMatch = code.match(/nn\.Linear\s*\(\s*(?:\w+\s*,\s*)?(\w+)/);
        if (linearMatch && linearMatch[1] === param.name) {
            // This param is an output dimension
            return '1024';
        }
        
        // Default
        return '1024';
    }
}










