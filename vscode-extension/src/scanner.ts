/**
 * AI-BOM scanner wrapper - executes the Python CLI and parses results
 */

import * as vscode from 'vscode';
import * as child_process from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import { ScanResult, ScannerConfig } from './types';

export class AIBOMScanner {
  private config: ScannerConfig;
  private outputChannel: vscode.OutputChannel;

  constructor(config: ScannerConfig, outputChannel: vscode.OutputChannel) {
    this.config = config;
    this.outputChannel = outputChannel;
  }

  /**
   * Validate that pythonPath does not contain shell metacharacters.
   * Prevents command injection via malicious Python path configuration.
   * OWASP: A03:2021 - Injection
   */
  private validatePythonPath(pythonPath: string): void {
    if (/[;&|`$(){}]/.test(pythonPath)) {
      throw new Error('Invalid Python path: contains shell metacharacters');
    }
  }

  /**
   * Check if ai-bom is installed
   */
  async isInstalled(): Promise<boolean> {
    try {
      this.validatePythonPath(this.config.pythonPath);
      const result = await this.executeCommand(this.config.pythonPath, ['-m', 'pip', 'show', 'ai-bom']);
      return result.exitCode === 0;
    } catch (error) {
      return false;
    }
  }

  /**
   * Install ai-bom via pip
   */
  async install(): Promise<boolean> {
    try {
      this.validatePythonPath(this.config.pythonPath);
      this.outputChannel.appendLine('Installing ai-bom...');
      const result = await this.executeCommand(
        this.config.pythonPath,
        ['-m', 'pip', 'install', 'ai-bom'],
        { timeout: 60000 }
      );

      if (result.exitCode === 0) {
        this.outputChannel.appendLine('ai-bom installed successfully');
        return true;
      } else {
        this.outputChannel.appendLine(`Installation failed: ${result.stderr}`);
        return false;
      }
    } catch (error) {
      this.outputChannel.appendLine(`Installation error: ${error}`);
      return false;
    }
  }

  /**
   * Scan a file or directory
   */
  async scan(targetPath: string): Promise<ScanResult> {
    // Check if ai-bom is installed
    const installed = await this.isInstalled();
    if (!installed) {
      if (this.config.autoInstall) {
        const installed = await this.install();
        if (!installed) {
          throw new Error('ai-bom is not installed. Install it with: pip install ai-bom');
        }
      } else {
        throw new Error('ai-bom is not installed. Install it with: pip install ai-bom');
      }
    }

    // Create temp file for JSON output
    const tempFile = path.join(
      fs.mkdtempSync(path.join(require('os').tmpdir(), 'ai-bom-')),
      'results.json'
    );

    try {
      // Build command
      const args = [
        'scan',
        targetPath,
        '--format',
        'cyclonedx',
        '--output',
        tempFile,
        '--quiet'
      ];

      if (this.config.deepScan) {
        args.push('--deep');
      }

      this.validatePythonPath(this.config.pythonPath);
      const execArgs = ['-m', 'ai_bom.cli', ...args];
      this.outputChannel.appendLine(`Running: ${this.config.pythonPath} ${execArgs.join(' ')}`);

      // Execute scan
      const result = await this.executeCommand(this.config.pythonPath, execArgs, { timeout: 120000 });

      if (result.exitCode !== 0 && result.exitCode !== 1) {
        // Exit code 1 is used for policy failures, which is okay
        throw new Error(`Scan failed: ${result.stderr}`);
      }

      // Read results
      const rawOutput = fs.readFileSync(tempFile, 'utf-8');
      const cyclonedxResult = JSON.parse(rawOutput);

      // Convert CycloneDX to our internal format
      const scanResult = this.convertCycloneDXToScanResult(cyclonedxResult, targetPath);

      this.outputChannel.appendLine(`Scan completed: ${scanResult.components.length} components found`);

      return scanResult;
    } finally {
      // Cleanup temp file
      try {
        if (fs.existsSync(tempFile)) {
          fs.unlinkSync(tempFile);
        }
      } catch (error) {
        // Ignore cleanup errors
      }
    }
  }

  /**
   * Convert CycloneDX format to our internal ScanResult format
   */
  private convertCycloneDXToScanResult(cyclonedx: any, targetPath: string): ScanResult {
    const components = (cyclonedx.components || []).map((comp: any) => {
      // Extract Trusera properties
      const props = comp.properties || [];
      const getRiskProp = (name: string): string => {
        const prop = props.find((p: any) => p.name === `trusera:ai-bom:${name}`);
        return prop?.value || '';
      };

      const riskScore = parseInt(getRiskProp('risk-score') || '0', 10);
      const severity = getRiskProp('severity') || 'low';
      const factors = getRiskProp('risk-factors')?.split(',') || [];
      const owaspCategories = getRiskProp('owasp-categories')?.split(',') || [];
      const filePath = getRiskProp('file-path') || '';
      const lineNumber = getRiskProp('line-number');
      const componentType = getRiskProp('component-type') || 'llm_provider';
      const provider = getRiskProp('provider') || '';
      const modelName = getRiskProp('model-name') || '';
      const flags = getRiskProp('flags')?.split(',') || [];

      return {
        id: comp.bom_ref || comp['bom-ref'] || `comp-${Math.random()}`,
        name: comp.name,
        type: componentType,
        version: comp.version || '',
        provider: provider,
        model_name: modelName,
        location: {
          file_path: filePath,
          line_number: lineNumber ? parseInt(lineNumber, 10) : null,
          context_snippet: ''
        },
        usage_type: 'unknown',
        risk: {
          score: riskScore,
          severity: severity as any,
          factors: factors,
          owasp_categories: owaspCategories
        },
        metadata: {},
        flags: flags,
        source: 'vscode-scan'
      };
    });

    // Build summary
    const bySeverity = components.reduce(
      (acc: any, comp: any) => {
        acc[comp.risk.severity] = (acc[comp.risk.severity] || 0) + 1;
        return acc;
      },
      { critical: 0, high: 0, medium: 0, low: 0 }
    );

    const byType = components.reduce((acc: any, comp: any) => {
      acc[comp.type] = (acc[comp.type] || 0) + 1;
      return acc;
    }, {});

    const highestRiskScore = components.reduce(
      (max: number, comp: any) => Math.max(max, comp.risk.score),
      0
    );

    return {
      target_path: targetPath,
      scan_timestamp: new Date().toISOString(),
      ai_bom_version: cyclonedx.metadata?.tools?.[0]?.version || 'unknown',
      components: components,
      summary: {
        total_components: components.length,
        total_files_scanned: 0,
        by_severity: bySeverity,
        by_type: byType,
        scan_duration_seconds: 0,
        highest_risk_score: highestRiskScore
      }
    };
  }

  /**
   * Execute a command safely using execFile (no shell injection).
   * Uses execFile instead of exec to avoid shell interpretation of arguments.
   * OWASP: A03:2021 - Injection
   */
  private executeCommand(
    executable: string,
    args: string[],
    options?: { timeout?: number }
  ): Promise<{ exitCode: number; stdout: string; stderr: string }> {
    return new Promise((resolve, reject) => {
      child_process.execFile(
        executable,
        args,
        {
          timeout: options?.timeout || 30000,
          maxBuffer: 10 * 1024 * 1024 // 10MB buffer
        },
        (error, stdout, stderr) => {
          if (error && !error.code) {
            reject(error);
          } else {
            resolve({
              exitCode: error?.code || 0,
              stdout: stdout.toString(),
              stderr: stderr.toString()
            });
          }
        }
      );
    });
  }
}
