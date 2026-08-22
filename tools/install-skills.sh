#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Incin AI Agent Skills Installer
# ==============================================================================
# Installs Incin agent skills into your local workspace or global AI editor config.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/xupremix/incin/master/tools/install-skills.sh | bash
#   bash tools/install-skills.sh [--tool <cursor|antigravity|claude|windsurf|all>] [--global]
# ==============================================================================

BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m'

TOOL="${1:-all}"
REPO_RAW_URL="https://raw.githubusercontent.com/xupremix/incin/master"

echo -e "${BOLD}${BLUE}🔥 Incin AI Agent Skills Installer${NC}"
echo -e "Installing Incin deep learning skills for AI coding assistants...\n"

install_skill() {
    local skill_name="$1"
    local dest_dir="$2"
    mkdir -p "$dest_dir/$skill_name"
    local dest_file="$dest_dir/$skill_name/SKILL.md"
    local src_file=".agents/skills/$skill_name/SKILL.md"
    
    # If in local repo and not writing to same file, copy; otherwise download from GitHub raw
    if [[ -f "$src_file" ]]; then
        if [[ "$(realpath -q "$src_file" 2>/dev/null || echo "")" != "$(realpath -q "$dest_file" 2>/dev/null || echo "")" ]]; then
            cp "$src_file" "$dest_file"
        fi
    else
        curl -fsSL "$REPO_RAW_URL/.agents/skills/$skill_name/SKILL.md" -o "$dest_file"
    fi
    echo -e "  ${GREEN}✓${NC} Installed ${BOLD}$skill_name${NC} -> $dest_file"
}

install_cursor_rule() {
    local skill_name="$1"
    local dest_dir="$2"
    mkdir -p "$dest_dir"
    local dest_file="$dest_dir/$skill_name.mdc"
    local src_file=".agents/skills/$skill_name/SKILL.md"
    
    if [[ -f "$src_file" ]]; then
        if [[ "$(realpath -q "$src_file" 2>/dev/null || echo "")" != "$(realpath -q "$dest_file" 2>/dev/null || echo "")" ]]; then
            cp "$src_file" "$dest_file"
        fi
    else
        curl -fsSL "$REPO_RAW_URL/.agents/skills/$skill_name/SKILL.md" -o "$dest_file"
    fi
    echo -e "  ${GREEN}✓${NC} Installed Cursor rule ${BOLD}$skill_name.mdc${NC} -> $dest_file"
}

SKILLS=("incin-expert" "incin-engineering" "incin-repository")

# Target directories based on requested tool
case "$TOOL" in
    antigravity|agy|gemini)
        echo -e "${YELLOW}Configuring for Google Antigravity / Gemini CLI (.agents/skills/)...${NC}"
        for s in "${SKILLS[@]}"; do install_skill "$s" ".agents/skills"; done
        ;;
    cursor)
        echo -e "${YELLOW}Configuring for Cursor (.cursor/rules/)...${NC}"
        for s in "${SKILLS[@]}"; do install_cursor_rule "$s" ".cursor/rules"; done
        ;;
    claude)
        echo -e "${YELLOW}Configuring for Claude Code (.claude/skills/)...${NC}"
        for s in "${SKILLS[@]}"; do install_skill "$s" ".claude/skills"; done
        ;;
    windsurf)
        echo -e "${YELLOW}Configuring for Windsurf (.windsurf/rules/)...${NC}"
        for s in "${SKILLS[@]}"; do install_cursor_rule "$s" ".windsurf/rules"; done
        ;;
    all|*)
        echo -e "${YELLOW}Configuring for standard agent directories (.agents/skills/ and .cursor/rules/)...${NC}"
        for s in "${SKILLS[@]}"; do
            install_skill "$s" ".agents/skills"
            install_cursor_rule "$s" ".cursor/rules"
        done
        ;;
esac

echo -e "\n${BOLD}${GREEN}Successfully installed Incin Agent Skills!${NC}"
echo -e "Your AI assistant is now equipped with Incin shape safety, target APIs, and architecture knowledge.\n"
