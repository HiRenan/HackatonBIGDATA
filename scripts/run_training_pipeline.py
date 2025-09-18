#!/usr/bin/env python3
"""
Script simplificado para treinamento do pipeline de modelos
Executa os principais modelos do PASSO 4 de forma sequencial e organizada
"""

import sys
import os
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    """Execução principal do pipeline de treinamento."""

    # Configurar encoding para Windows
    if os.name == 'nt':  # Windows
        try:
            os.system('chcp 65001 > nul 2>&1')  # UTF-8
        except:
            pass

    print("="*60)
    print("PIPELINE DE TREINAMENTO - MODELOS HACKATHON 2025")
    print("="*60)
    print("Iniciando treinamento sequencial dos modelos...")

    models_to_run = [
        {
            'name': 'Prophet Seasonal',
            'module': 'src.models.prophet_seasonal',
            'description': 'Modelo Prophet com sazonalidade avançada'
        },
        {
            'name': 'LightGBM Master',
            'module': 'src.models.lightgbm_master',
            'description': 'LightGBM otimizado para WMAPE'
        },
        {
            'name': 'Advanced Ensemble',
            'module': 'src.models.advanced_ensemble',
            'description': 'Sistema de ensemble multicamadas'
        }
    ]

    successful_models = []
    failed_models = []

    for i, model in enumerate(models_to_run, 1):
        print(f"\n[{i}/{len(models_to_run)}] Executando {model['name']}...")
        print(f"Descrição: {model['description']}")
        print("-" * 50)

        try:
            # Importar e executar o módulo
            exec(f"import {model['module']}")
            print(f"[OK] {model['name']} executado com sucesso!")
            successful_models.append(model['name'])

        except ImportError as e:
            print(f"[ERRO] Erro de importacao para {model['name']}: {e}")
            failed_models.append(model['name'])

        except Exception as e:
            print(f"[ERRO] Erro na execucao de {model['name']}: {e}")
            failed_models.append(model['name'])

    # Relatório final
    print(f"\n{'='*60}")
    print("RELATÓRIO FINAL DO PIPELINE")
    print(f"{'='*60}")
    print(f"Modelos executados com sucesso: {len(successful_models)}")
    for model in successful_models:
        print(f"  [OK] {model}")

    if failed_models:
        print(f"\nModelos com problemas: {len(failed_models)}")
        for model in failed_models:
            print(f"  [ERRO] {model}")

    print(f"\nTaxa de sucesso: {len(successful_models)}/{len(models_to_run)} ({100*len(successful_models)/len(models_to_run):.1f}%)")

    if len(successful_models) == len(models_to_run):
        print("\n[SUCESSO] PIPELINE COMPLETO! Todos os modelos foram executados com sucesso.")
        return 0
    else:
        print(f"\n[AVISO] Pipeline incompleto. Revisar modelos com problemas.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)