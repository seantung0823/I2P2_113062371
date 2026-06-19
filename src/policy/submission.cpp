#include <utility>
#include "state.hpp"
#include "submission.hpp"

/*============================================================
 * MiniMax — eval_ctx
 *
 * Alpha-Beta Pruning version.
 * 你的程式原本是 Negamax，所以不需要 maximizingPlayer。
 * alpha = 目前已知最好下界
 * beta  = 對手能接受的上界
 *============================================================*/
int submission::eval_ctx(
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

    history.push(state->hash());

    if(depth <= 0){
        int score = state->evaluate(
            p.use_kp_eval,
            p.use_eval_mobility,
            &history
        );

        history.pop(state->hash());
        return score;
    }

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
             * 用正常 Alpha-Beta window 搜尋
             */
            if(same){
                raw = eval_ctx( next, depth - 1, history, ply + 1, ctx, p, alpha, beta);
                score = raw;
            }else{
                raw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -alpha);
                score = -raw;
            }

            first_child = false;
        }else{
            /*
             * 後面的 child：
             * 先用窄視窗搜尋
             */
            if(same){
                raw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, alpha, alpha + 1 );
                score = raw;
            }else{
                raw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -(alpha + 1), -alpha );
                score = -raw;
            }

            /*
             * 如果窄視窗搜尋發現：
             * 這個 child 有可能比目前最好還好，
             * 才重新用完整 window 搜一次。
             */
            if(score > alpha && score < beta){
                if(same){
                    raw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, alpha, beta );
                    score = raw;
                }else{
                    raw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -alpha );
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

    bool first_child = true;

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);

        bool same = next->same_player_as_parent();

        int raw;
        int score;

        if(first_child){
            /*
             * 第一個 root move：
             * 用正常 Alpha-Beta window 搜尋
             */
            if(same){
                raw = eval_ctx(next, depth - 1, history, 1, ctx, p, alpha, beta );
                score = raw;
            }else{
                raw = eval_ctx(next, depth - 1, history, 1, ctx, p, -beta, -alpha );
                score = -raw;
            }

            first_child = false;
        }else{
            /*
             * 後面的 root move：
             * 先用窄視窗搜尋
             */
            if(same){
                raw = eval_ctx( next, depth - 1, history, 1, ctx, p, alpha, alpha + 1 );
                score = raw;
            }else{
                raw = eval_ctx( next, depth - 1, history, 1, ctx, p, -(alpha + 1), -alpha );
                score = -raw;
            }

            /*
             * 如果這步可能比目前最佳解更好，
             * 才重新完整搜尋。
             */
            if(score > alpha && score < beta){
                if(same){
                    raw = eval_ctx( next, depth - 1, history, 1, ctx, p, alpha, beta );
                    score = raw;
                }else{
                    raw = eval_ctx( next, depth - 1, history, 1, ctx, p, -beta, -alpha );
                    score = -raw;
                }
            }
        }

        delete next;

        if(score > best_score){
            best_score = score;
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
