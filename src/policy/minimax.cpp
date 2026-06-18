#include <utility>
#include "state.hpp"
#include "minimax.hpp"


/*============================================================
 * MiniMax — eval_ctx
 *
 * Alpha-Beta Pruning version.
 * 你的程式原本是 Negamax，所以不需要 maximizingPlayer。
 * alpha = 目前已知最好下界
 * beta  = 對手能接受的上界
 *============================================================*/
int MiniMax::eval_ctx(
    State *state,
    int depth,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p,
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

    /* === Lazy move generation (sets game_state) === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal / leaf checks === */

    // 勝利局面：越快贏越好，所以扣 ply
    if(state->game_state == WIN){
        return P_MAX - ply;
    }

    if(state->game_state == DRAW){
        return 0;
    }

    /* === Repetition check (game-specific) === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }
    history.push(state->hash());

    if(depth <= 0){
        int score = state->evaluate(
            p.use_kp_eval, p.use_eval_mobility, &history
        );
        history.pop(state->hash());
        return score;
    }

    /* === Alpha-Beta loop === */
    int best_score = M_MAX;

    for(auto& action : state->legal_actions){
        State *next = state->next_state(action);

        bool same = next->same_player_as_parent();

        int raw;
        int score;

        if(same){
            // 如果下一層還是同一個玩家，分數方向不用反轉
            raw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, alpha, beta);
            score = raw;
        }else{
            // 如果換對手走，分數方向要反轉，所以 alpha/beta 也要反過來
            raw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -alpha);
            score = -raw;
        }

        delete next;

        // value := max(value, child_score)
        if(score > best_score){
            best_score = score;
        }

        // alpha := max(alpha, value)
        if(best_score > alpha){
            alpha = best_score;
        }

        // if alpha >= beta then break
        // 這就是 beta cutoff：後面的 child 不用看了
        if(alpha >= beta){
            break;
        }
    }

    history.pop(state->hash());
    return best_score;
}


/*============================================================
 * MiniMax — search
 *
 * Iterate legal moves, call eval_ctx, return SearchResult.
 *============================================================*/
SearchResult MiniMax::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();
    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }

    int best_score = M_MAX;
    int alpha = M_MAX;
    int beta = P_MAX;

    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);

        bool same = next->same_player_as_parent();

        int raw;
        int score;

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

        delete next;

        if(score > best_score){
            best_score = score;
            result.best_move = action;

            if(p.report_partial && ctx.on_root_update){
                ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
            }
        }

        // root 這層也要更新 alpha，後面的 move 才能剪枝
        if(best_score > alpha){
            alpha = best_score;
        }

        // 通常 beta 一開始是 P_MAX，所以 root 不太會剪，
        // 但保留這行是完整的 Alpha-Beta 寫法。
        if(alpha >= beta){
            break;
        }

        move_index++;
    }

    result.score = best_score;
    result.nodes = ctx.nodes;
    result.seldepth = ctx.seldepth;

    return result;
}


/*============================================================
 * MiniMax — default_params / param_defs
 *============================================================*/
ParamMap MiniMax::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "true"},
    };
}

std::vector<ParamDef> MiniMax::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
    };
}
