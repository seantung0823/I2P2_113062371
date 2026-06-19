#include <utility>
#include <iostream>
#include <algorithm>
#include <vector>
#include "state.hpp"
#include "submission.hpp"

/*============================================================
 * Quiescence Search 設定
 *
 * 只在 depth == 0 之後繼續搜尋「吃子 / 升變」
 * 避免剛好停在一個很不穩定的局面。
 *
 * MAX_Q_DEPTH 不要設太大，不然會超時。
 *============================================================*/
static const int MAX_Q_DEPTH = 4;


/*============================================================
 * 判斷這步是不是吃子
 *
 * Move = pair<Point, Point>
 * action.first  = from
 * action.second = to
 *
 * 如果目標格有對手棋子，就是 capture。
 *============================================================*/
static bool is_capture(State *state, const Move& action){
    Point to = action.second;

    int me = state->player;
    int opp = 1 - me;

    return state->board.board[opp][to.first][to.second] != 0;
}


/*============================================================
 * 判斷這步是不是兵升變
 *
 * Pawn 到最後一排會變 Queen。
 *============================================================*/
static bool is_promotion(State *state, const Move& action){
    Point from = action.first;
    Point to = action.second;

    int me = state->player;
    int piece = state->board.board[me][from.first][from.second];

    if(piece != 1){
        return false;
    }

    return to.first == 0 || to.first == BOARD_H - 1;
}


/*============================================================
 * Quiescence 只搜尋 noisy move
 *
 * noisy move:
 * 1. 吃子
 * 2. 升變
 *============================================================*/
static bool is_noisy(State *state, const Move& action){
    return is_capture(state, action) || is_promotion(state, action);
}


/*============================================================
 * Quiescence Search
 *
 * stand_pat：
 *   目前盤面直接 evaluate 的分數。
 *
 * 如果目前分數已經 >= beta：
 *   對手不會讓你走到這裡，可以直接 cutoff。
 *
 * 如果還沒 cutoff：
 *   只繼續搜尋吃子 / 升變。
 *============================================================*/
static int quiescence(
    State *state,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const SubmissionParams& p,
    int alpha,
    int beta,
    int q_depth
){
    ctx.nodes++;

    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }

    if(ctx.stop){
        return 0;
    }

    /* === Lazy move generation === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal checks === */
    if(state->game_state == WIN){
        return P_MAX - ply;
    }

    if(state->game_state == DRAW){
        return 0;
    }

    /* === Repetition check === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }

    history.push(state->hash());

    /* === Stand pat evaluation === */
    int stand_pat = state->evaluate(
        p.use_kp_eval,
        p.use_eval_mobility,
        &history
    );

    /*
     * 如果現在不走棋就已經好到 beta 以上，
     * 代表這條線對手不會選，直接剪枝。
     */
    if(stand_pat >= beta){
        history.pop(state->hash());
        return beta;
    }

    /*
     * 目前盤面比 alpha 好，就更新 alpha。
     */
    if(stand_pat > alpha){
        alpha = stand_pat;
    }

    /*
     * 限制 Quiescence 深度，避免吃子鏈太長造成超時。
     */
    if(q_depth <= 0){
        history.pop(state->hash());
        return alpha;
    }

    /* === 只搜尋吃子 / 升變 === */
    for(auto& action : state->legal_actions){

        if(!is_noisy(state, action)){
            continue;
        }

        State *next = state->next_state(action);

        bool same = next->same_player_as_parent();

        int raw;
        int score;

        if(same){
            raw = quiescence(
                next,
                history,
                ply + 1,
                ctx,
                p,
                alpha,
                beta,
                q_depth - 1
            );
            score = raw;
        }else{
            raw = quiescence(
                next,
                history,
                ply + 1,
                ctx,
                p,
                -beta,
                -alpha,
                q_depth - 1
            );
            score = -raw;
        }

        delete next;

        if(score > alpha){
            alpha = score;
        }

        if(alpha >= beta){
            break;
        }
    }

    history.pop(state->hash());
    return alpha;
}


/*============================================================
 * MiniMax — eval_ctx
 *
 * PVS + Alpha-Beta + Quiescence version.
 *============================================================*/
/*============================================================
 * Move ordering 強化版
 *
 * 不改 PVS / Quiescence 的核心，只讓「比較可能好的走法」
 * 排在前面，讓 PVS + Alpha-Beta 更容易剪枝、看更深。
 *============================================================*/
static const int MAX_KILLER_PLY = 128;
static Move killer_1[MAX_KILLER_PLY];
static Move killer_2[MAX_KILLER_PLY];
static bool has_killer_1[MAX_KILLER_PLY] = {false};
static bool has_killer_2[MAX_KILLER_PLY] = {false};

static bool same_move(const Move& a, const Move& b){
    return a.first.first == b.first.first &&
           a.first.second == b.first.second &&
           a.second.first == b.second.first &&
           a.second.second == b.second.second;
}

static int piece_value(int piece){
    switch(piece){
        case 1: return 100;    // Pawn
        case 2: return 320;    // Knight
        case 3: return 330;    // Bishop
        case 4: return 500;    // Rook
        case 5: return 900;    // Queen
        case 6: return 20000;  // King
        default: return 0;
    }
}

/*
 * 越靠近中心越好。
 * 這只是排序用的小分數，不會覆蓋搜尋結果。
 */
static int center_score(const Point& p){
    int r2 = 2 * p.first - (BOARD_H - 1);
    int c2 = 2 * p.second - (BOARD_W - 1);

    if(r2 < 0) r2 = -r2;
    if(c2 < 0) c2 = -c2;

    return 30 - 4 * (r2 + c2);
}

static void save_killer(int ply, const Move& action){
    if(ply < 0 || ply >= MAX_KILLER_PLY){
        return;
    }

    if(has_killer_1[ply] && same_move(killer_1[ply], action)){
        return;
    }

    killer_2[ply] = killer_1[ply];
    has_killer_2[ply] = has_killer_1[ply];

    killer_1[ply] = action;
    has_killer_1[ply] = true;
}

static int move_score(State *state, const Move& action, int ply){
    Point from = action.first;
    Point to = action.second;

    int me = state->player;
    int opp = 1 - me;

    int moving_piece = state->board.board[me][from.first][from.second];
    int captured_piece = state->board.board[opp][to.first][to.second];

    int score = 0;

    /*
     * 1. 吃王最高：這種 move 一定要最前面。
     */
    if(captured_piece == 6){
        score += 10000000;
    }

    /*
     * 2. MVV-LVA：
     * Most Valuable Victim - Least Valuable Attacker
     * 優先：用便宜的子吃貴的子。
     */
    if(captured_piece != 0){
        score += 100000;
        score += piece_value(captured_piece) * 20;
        score -= piece_value(moving_piece);
    }

    /*
     * 3. 升變很重要，排前面。
     */
    if(moving_piece == 1 && (to.first == 0 || to.first == BOARD_H - 1)){
        score += 80000;
    }

    /*
     * 4. killer move：
     * 如果同一層之前某個非吃子 move 造成 cutoff，
     * 之後同一層優先試它。
     */
    if(ply >= 0 && ply < MAX_KILLER_PLY){
        if(has_killer_1[ply] && same_move(killer_1[ply], action)){
            score += 70000;
        }else if(has_killer_2[ply] && same_move(killer_2[ply], action)){
            score += 60000;
        }
    }

    /*
     * 5. 小分數：中心控制。
     * 這不會決定勝負，只是同分時讓 move order 比較合理。
     */
    score += center_score(to);

    /*
     * 6. 兵往底線附近走，稍微加分。
     */
    if(moving_piece == 1){
        int dist_top = to.first;
        int dist_bottom = BOARD_H - 1 - to.first;
        int near_promotion = dist_top < dist_bottom ? dist_top : dist_bottom;
        score += (BOARD_H - 1 - near_promotion) * 10;
    }

    return score;
}

static void order_moves(State *state, int ply){
    std::vector< std::pair<int, Move> > scored;

    for(auto& action : state->legal_actions){
        scored.push_back(std::make_pair(move_score(state, action, ply), action));
    }

    std::stable_sort(
        scored.begin(),
        scored.end(),
        [](const std::pair<int, Move>& a, const std::pair<int, Move>& b){
            return a.first > b.first;
        }
    );

    for(int i = 0; i < (int)scored.size(); i++){
        state->legal_actions[i] = scored[i].second;
    }
}

int submission::eval_ctx(
    State *state,
    int depth,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const SubmissionParams& p,
    int alpha,
    int beta
){
    ctx.nodes++;

    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }

    if(ctx.stop){
        return 0;
    }

    /* === Lazy move generation === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal / leaf checks === */
    if(state->game_state == WIN){
        return P_MAX - ply;
    }

    if(state->game_state == DRAW){
        return 0;
    }

    /* === Repetition check === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }

    /*
     * 重點：
     * depth 到 0 時，不再直接 evaluate，
     * 而是進入 Quiescence Search。
     *
     * 注意這裡還沒 history.push，
     * 因為 quiescence 裡面自己會 push/pop。
     */
    if(depth <= 0){
        return quiescence(
            state,
            history,
            ply,
            ctx,
            p,
            alpha,
            beta,
            MAX_Q_DEPTH
        );
    }

    history.push(state->hash());

    order_moves(state, ply);

    /* === PVS loop === */
    int best_score = M_MAX;
    bool first_child = true;

    for(auto& action : state->legal_actions){
        State *next = state->next_state(action);

        bool same = next->same_player_as_parent();

        int raw;
        int score;

        if(first_child){
            /*
             * 第一個 child：
             * 用正常 Alpha-Beta window 搜尋。
             */
            if(same){
                raw = eval_ctx(
                    next,
                    depth - 1,
                    history,
                    ply + 1,
                    ctx,
                    p,
                    alpha,
                    beta
                );
                score = raw;
            }else{
                raw = eval_ctx(
                    next,
                    depth - 1,
                    history,
                    ply + 1,
                    ctx,
                    p,
                    -beta,
                    -alpha
                );
                score = -raw;
            }

            first_child = false;
        }else{
            /*
             * 後面的 child：
             * 先用窄視窗搜尋。
             */
            if(same){
                raw = eval_ctx(
                    next,
                    depth - 1,
                    history,
                    ply + 1,
                    ctx,
                    p,
                    alpha,
                    alpha + 1
                );
                score = raw;
            }else{
                raw = eval_ctx(
                    next,
                    depth - 1,
                    history,
                    ply + 1,
                    ctx,
                    p,
                    -(alpha + 1),
                    -alpha
                );
                score = -raw;
            }

            /*
             * 如果窄視窗搜尋發現：
             * 這個 child 有可能比目前最好還好，
             * 才重新用完整 window 搜一次。
             */
            if(score > alpha && score < beta){
                if(same){
                    raw = eval_ctx(
                        next,
                        depth - 1,
                        history,
                        ply + 1,
                        ctx,
                        p,
                        alpha,
                        beta
                    );
                    score = raw;
                }else{
                    raw = eval_ctx(
                        next,
                        depth - 1,
                        history,
                        ply + 1,
                        ctx,
                        p,
                        -beta,
                        -alpha
                    );
                    score = -raw;
                }
            }
        }

        delete next;

        if(score > best_score){
            best_score = score;
        }

        if(best_score > alpha){
            alpha = best_score;
        }

        if(alpha >= beta){
            /*
             * 這個 action 造成 cutoff，記成 killer move。
             * 只記 quiet move，吃子本來就已經會排很前面。
             */
            if(!is_noisy(state, action)){
                save_killer(ply, action);
            }
            break;
        }
    }

    history.pop(state->hash());
    return best_score;
}


/*============================================================
 * MiniMax — search
 *
 * Root search with PVS.
 *============================================================*/
SearchResult submission::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();

    SubmissionParams p = SubmissionParams::from_map(ctx.params);

    /*
     * 額外多看 1 層：
     * 這是最直接讓 AI 變強的地方。
     * 如果你跑起來超時，就把 EXTRA_SEARCH_DEPTH 改成 0。
     */
    static const int EXTRA_SEARCH_DEPTH = 0;
    if(depth > 0){
        depth += EXTRA_SEARCH_DEPTH;
    }

    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }

    if(state->game_state == WIN){
        result.best_move = state->legal_actions.back();
        result.score = P_MAX;
        result.nodes = ctx.nodes;
        result.seldepth = ctx.seldepth;
        return result;
    }

    if(state->legal_actions.empty()){
        result.score = 0;
        result.nodes = ctx.nodes;
        result.seldepth = ctx.seldepth;
        return result;
    }

    order_moves(state, 0);

    int best_score = M_MAX;
    int alpha = M_MAX;
    int beta = P_MAX;

    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();

    bool first_child = true;
    int best_order_score = M_MAX;

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);

        bool same = next->same_player_as_parent();

        int raw;
        int score;

        if(first_child){
            /*
             * 第一個 root move：
             * 用正常 Alpha-Beta window 搜尋。
             */
            if(same){
                raw = eval_ctx(
                    next,
                    depth - 1,
                    history,
                    1,
                    ctx,
                    p,
                    alpha,
                    beta
                );
                score = raw;
            }else{
                raw = eval_ctx(
                    next,
                    depth - 1,
                    history,
                    1,
                    ctx,
                    p,
                    -beta,
                    -alpha
                );
                score = -raw;
            }

            first_child = false;
        }else{
            /*
             * 後面的 root move：
             * 先用窄視窗搜尋。
             */
            if(same){
                raw = eval_ctx(
                    next,
                    depth - 1,
                    history,
                    1,
                    ctx,
                    p,
                    alpha,
                    alpha + 1
                );
                score = raw;
            }else{
                raw = eval_ctx(
                    next,
                    depth - 1,
                    history,
                    1,
                    ctx,
                    p,
                    -(alpha + 1),
                    -alpha
                );
                score = -raw;
            }

            /*
             * 如果這步可能比目前最佳解更好，
             * 才重新完整搜尋。
             */
            if(score > alpha && score < beta){
                if(same){
                    raw = eval_ctx(
                        next,
                        depth - 1,
                        history,
                        1,
                        ctx,
                        p,
                        alpha,
                        beta
                    );
                    score = raw;
                }else{
                    raw = eval_ctx(
                        next,
                        depth - 1,
                        history,
                        1,
                        ctx,
                        p,
                        -beta,
                        -alpha
                    );
                    score = -raw;
                }
            }
        }

        delete next;

        int this_order_score = move_score(state, action, 0);

        if(score > best_score || (score == best_score && this_order_score > best_order_score)){
            best_score = score;
            best_order_score = this_order_score;
            result.best_move = action;

            if(p.report_partial && ctx.on_root_update){
                ctx.on_root_update({
                    result.best_move,
                    best_score,
                    depth,
                    move_index + 1,
                    total_moves
                });
            }
        }

        if(best_score > alpha){
            alpha = best_score;
        }

        if(alpha >= beta){
            break;
        }

        move_index++;
    }

    result.score = best_score;
    result.nodes = ctx.nodes;
    result.seldepth = ctx.seldepth;
    result.pv = {result.best_move};

    return result;
}


/*============================================================
 * MiniMax — default_params / param_defs
 *============================================================*/
ParamMap submission::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "true"},
    };
}

std::vector<ParamDef> submission::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
    };
}
