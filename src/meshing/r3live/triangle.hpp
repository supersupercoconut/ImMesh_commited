#pragma once
#include <set>
#include <unordered_set>
#include "tools_kd_hash.hpp"
#include "pointcloud_rgbd.hpp"
// class RGB_pts;
// class RGB_Voxel;
class Global_map;
extern std::mutex g_mutex_sync_triangle_set;

// 三角形mesh面片的基本结构
class Triangle
{
  public:
    // m_tri_pts_id 这个数据难道对应的是在triangle上面的id ?
    int    m_tri_pts_id[ 3 ] = { 0 };
    vec_3  m_normal = vec_3( 0, 0, 0 );
    int    m_projected_texture_id = 0;
    vec_2f m_texture_coor[ 3 ];     // 这里对应的应该是三个顶点对应的图像像素位置(也就是进行颜色配置)
    double m_vis_score = 0;
    float  last_texture_distance = 3e8;
    int    m_index_flip = 0;

    /////////// 新增数据 ////////////
    double m_tri_pts_color[3][3];   // 对应三个顶点的颜色信息
    int m_is_colored = false;
    ///////////////////////////////


    void sort_id()
    {
        std::sort( std::begin( m_tri_pts_id ), std::end( m_tri_pts_id ) );
    }
    Triangle() = default;
    ~Triangle() = default;

    Triangle( int id_0, int id_1, int id_2 )
    {
        m_tri_pts_id[ 0 ] = id_0;
        m_tri_pts_id[ 1 ] = id_1;
        m_tri_pts_id[ 2 ] = id_2;

        /// 补充:
        double m_tri_pts_color[3][3] = {
                {0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0}
        };

        sort_id();
    }
};

using Triangle_ptr = std::shared_ptr< Triangle >;
// using Triangle_set = std::unordered_set<Triangle_ptr>;
using Triangle_set = std::set< Triangle_ptr >;

// 需要被同步更新的三角部分
struct Sync_triangle_set
{
    Triangle_set m_triangle_set;
    std::shared_ptr< std::mutex> m_mutex_ptr = nullptr;
    int          m_frame_idx = 0;
    bool         m_if_required_synchronized = true;
    Sync_triangle_set() 
    {
//        m_mutex_ptr = std::make_shared<std::mutex>();
    }

    void lock()
    {
//        m_mutex_ptr->lock();
        g_mutex_sync_triangle_set.lock();
    }

    void unlock()
    {
//        m_mutex_ptr->unlock();
        g_mutex_sync_triangle_set.unlock();
    }

    void insert( const Triangle_ptr& tri_ptr )
    {
        lock();
        m_triangle_set.insert( tri_ptr );
        m_if_required_synchronized = true;
        unlock();
    }

    void erase( const Triangle_ptr& tri_ptr )
    {
        lock();
        auto it1 = m_triangle_set.find( tri_ptr );
        if ( it1 != m_triangle_set.end() )
        {
            m_triangle_set.erase( it1 );
        }
        m_if_required_synchronized = true;
        unlock();
    }

    void clear()
    {
        lock();
        m_triangle_set.clear();
        m_if_required_synchronized = true;
        unlock();
    }

    int get_triangle_set_size()
    {
        int return_size = 0;
        lock();
        return_size = m_triangle_set.size();
        unlock();
        return return_size;
    }

    Triangle_set * get_triangle_set_ptr()
    {
        return &m_triangle_set;
    }
    
    void get_triangle_set(Triangle_set & ret_set, bool reset_status = false)
    {
        lock();
        ret_set = m_triangle_set;
        if ( reset_status )
        {
            m_if_required_synchronized = false;
        }
        unlock();
    }
};

class Triangle_manager
{
  public:
    Global_map*                             m_pointcloud_map = nullptr;
    // 当前所有的triangle都在这里被保存 - 方便快速查找一个triangle是不是已经存在
    Hash_map_3d< int, Triangle_ptr >        m_triangle_hash;

    double                                  m_region_size = 10.0;
    // m_triangle_set_vector 保存的是 每一个region中的 Sync_triangle_set
    std::vector< Sync_triangle_set* >            m_triangle_set_vector;
    // Hash_map_3d< int, Triangle_set >        m_triangle_set_in_region;
    Hash_map_3d< int, Sync_triangle_set >        m_triangle_set_in_region;

    std::unordered_map< int, Triangle_set > m_map_pt_triangle;

    // Triangle_set m_triangle_list;

    std::mutex m_mutex_triangle_hash;

    Hash_map_2d< int, Triangle_set > m_map_edge_triangle;
    Hash_map_2d< int, Triangle_set > m_map_edge_triangle_conflicted;
    int                              m_enable_map_edge_triangle = 0;
    int                              m_newest_rec_frame_idx = 0;
    
    void clear()
    {
        m_triangle_hash.clear();
        m_map_pt_triangle.clear();
        // m_triangle_list.clear();
        for ( auto& triangle_set : m_triangle_set_in_region.m_map_3d_hash_map )
        {
            triangle_set.second.clear();
        }

        m_map_edge_triangle.clear();
        m_map_edge_triangle_conflicted.clear();
    }

    Triangle_manager()
    {
        m_triangle_hash.m_map_3d_hash_map.reserve( 1e7 );
        m_map_pt_triangle.reserve( 1e7 );
    };
    ~Triangle_manager()
    {
        // 将new对应的部分进行释放
        for(auto* ptr : m_triangle_set_vector)
        {
            delete ptr;
        }
        m_triangle_set_vector.clear();
        m_triangle_set_in_region.clear();
    }

    vec_3 get_triangle_center( const Triangle_ptr& tri_ptr );
    void  insert_triangle_to_list( const Triangle_ptr& tri_ptr, const int& frame_idx = 0 );
    void  erase_triangle_from_list( const Triangle_ptr& tri_ptr, const int& frame_idx = 0 );
    int   get_all_triangle_list( std::vector< Triangle_set >& triangle_list, std::mutex* mutex = nullptr, int sleep_us_each_query = 10 );
    int   get_triangle_list_size();
    // void 

    void erase_triangle( const Triangle_ptr& tri_ptr )
    {
        int idx[ 3 ];
        idx[ 0 ] = tri_ptr->m_tri_pts_id[ 0 ];
        idx[ 1 ] = tri_ptr->m_tri_pts_id[ 1 ];
        idx[ 2 ] = tri_ptr->m_tri_pts_id[ 2 ];
        // printf_line;
        // erase triangle in list
        erase_triangle_from_list( tri_ptr, m_newest_rec_frame_idx );
        // auto it1 = m_triangle_list.find( tri_ptr );
        // if ( it1 != m_triangle_list.end() )
        // {
        //     m_triangle_list.erase( m_triangle_list.find( tri_ptr ) );
        // }

        for ( int tri_idx = 0; tri_idx < 3; tri_idx++ )
        {
            auto it3 = m_map_pt_triangle[ idx[ tri_idx ] ].find( tri_ptr );
            if ( it3 != m_map_pt_triangle[ idx[ tri_idx ] ].end() )
            {
                m_map_pt_triangle[ idx[ tri_idx ] ].erase( it3 );
            }
        }

        if ( m_enable_map_edge_triangle )
        {
            // printf_line;
            // erase triangle in edge-triangle list
            auto it2 = m_map_edge_triangle.m_map_2d_hash_map[ idx[ 0 ] ][ idx[ 1 ] ].find( tri_ptr );
            if ( it2 != m_map_edge_triangle.m_map_2d_hash_map[ idx[ 0 ] ][ idx[ 1 ] ].end() )
            {
                m_map_edge_triangle.m_map_2d_hash_map[ idx[ 0 ] ][ idx[ 1 ] ].erase( it2 );
            }

            it2 = m_map_edge_triangle.m_map_2d_hash_map[ idx[ 0 ] ][ idx[ 2 ] ].find( tri_ptr );
            if ( it2 != m_map_edge_triangle.m_map_2d_hash_map[ idx[ 0 ] ][ idx[ 2 ] ].end() )
            {
                m_map_edge_triangle.m_map_2d_hash_map[ idx[ 0 ] ][ idx[ 2 ] ].erase( it2 );
            }

            it2 = m_map_edge_triangle.m_map_2d_hash_map[ idx[ 1 ] ][ idx[ 2 ] ].find( tri_ptr );
            if ( it2 != m_map_edge_triangle.m_map_2d_hash_map[ idx[ 1 ] ][ idx[ 2 ] ].end() )
            {
                m_map_edge_triangle.m_map_2d_hash_map[ idx[ 1 ] ][ idx[ 2 ] ].erase( it2 );
            }
        }
        // printf_line;
    }

    void remove_triangle_list( const Triangle_set& triangle_list, const int frame_idx = 0 )
    {
        m_mutex_triangle_hash.lock();
        m_newest_rec_frame_idx = std::max( frame_idx, m_newest_rec_frame_idx );
        for ( auto tri_ptr : triangle_list )
        {
            erase_triangle( tri_ptr );
        }
        m_mutex_triangle_hash.unlock();
    }

    template < typename T >
    Triangle_set find_relative_triangulation_combination( std::set< T >& set_index )
    {
        // std::set< T >::iterator it;
        Triangle_set triangle_ptr_list;
        // m_mutex_triangle_hash.lock();

        // 根据输入序号进行分析
        for ( typename std::set< T >::iterator it = set_index.begin(); it != set_index.end(); it++ )
        {
            // 点存在对应的三角形
            if ( m_map_pt_triangle.find( *it ) != m_map_pt_triangle.end() )
            {
                // 找到这个点对应的三角形，三角形的三个点需要都属于输入的点集合中, 才认为这是一个合格的点 | tri_it对应的为triangle_set | 这里可能是一个点与多个triangle相链接,所以这里对应的是begin以及end
                for ( Triangle_set::iterator tri_it = m_map_pt_triangle[ *it ].begin(); tri_it != m_map_pt_triangle[ *it ].end(); tri_it++ )
                {
                    //  反向验证 证明三个点都应该属于triangle
                    if ( ( set_index.find( ( *tri_it )->m_tri_pts_id[ 0 ] ) != set_index.end() ) &&
                         ( set_index.find( ( *tri_it )->m_tri_pts_id[ 1 ] ) != set_index.end() ) &&
                         ( set_index.find( ( *tri_it )->m_tri_pts_id[ 2 ] ) != set_index.end() ) )
                    {

                        triangle_ptr_list.insert( *tri_it );
                    }
                }
            }
        }
        // m_mutex_triangle_hash.unlock();
        return triangle_ptr_list;
    }

    template < typename T >
    void remove_relative_triangulation_combination( std::set< T >& set_index )
    {
        // std::set< T >::iterator it;
        Triangle_set triangle_ptr_list = find_relative_triangulation_combination( set_index );

        // cout << ANSI_COLOR_YELLOW_BOLD << "In conflict triangle size = " << triangle_ptr_list.size() << ANSI_COLOR_RESET << endl;
        remove_triangle_list( triangle_ptr_list );
    }

    template < typename T >
    void remove_relative_triangulation_combination( std::vector< T >& vector_index )
    {
        std::set< T > index_set;
        for ( auto p : vector_index )
        {
            index_set.insert( p );
        }
        remove_relative_triangulation_combination( index_set );
    }

    template < typename T >
    Triangle_set get_inner_hull_triangle_list( std::set< T >& inner_hull_indices )
    {
        Triangle_set triangle_list;
        for ( auto p : inner_hull_indices )
        {
            if ( m_map_pt_triangle.find( p ) != m_map_pt_triangle.end() )
            {
                for ( auto pp : m_map_pt_triangle[ p ] )
                {
                    triangle_list.insert( pp );
                }
            }
        }
        return triangle_list;
    }

    template < typename T >
    void remove_inner_hull_triangle( std::set< T >& inner_hull_indices )
    {
        Triangle_set triangle_list = get_inner_hull_triangle_list( inner_hull_indices );
        remove_triangle_list( triangle_list );
    }

    bool if_triangle_exist( int& id_0, int& id_1, int& id_2 )
    {
        int ids[ 3 ];
        ids[ 0 ] = id_0;
        ids[ 1 ] = id_1;
        ids[ 2 ] = id_2;
        std::sort( std::begin( ids ), std::end( ids ) );
        if ( m_triangle_hash.if_exist( ids[ 0 ], ids[ 1 ], ids[ 2 ] ) )
        {
            // This triangle exist
            return true;
        }
        else
        {
            return false;
        }
    }

//    Triangle_ptr find_triangle( int id_0, int id_1, int id_2 )
//    {
//        int ids[ 3 ];
//        ids[ 0 ] = id_0;
//        ids[ 1 ] = id_1;
//        ids[ 2 ] = id_2;
//        std::sort( std::begin( ids ), std::end( ids ) );
//        if ( m_triangle_hash.if_exist( ids[ 0 ], ids[ 1 ], ids[ 2 ] ) )
//        {
//            // This triangle exist
//            // return m_triangle_hash.m_map_3d_hash_map[ ids[ 0 ] ][ ids[ 1 ] ][ ids[ 2 ] ];
//            return *m_triangle_hash.get_data( ids[ 0 ], ids[ 1 ], ids[ 2 ] );
//        }
//        else
//        {
//            return nullptr;
//        }
//    }

    // 定义一个数组参数, 用于获取颜色信息
//    Triangle_ptr insert_triangle( int id_0, int id_1, int id_2, double color[3][3],  int build_triangle_map = false, const int & frame_idx = 0 )
//    {
//        int ids[ 3 ];
//        ids[ 0 ] = id_0;
//        ids[ 1 ] = id_1;
//        ids[ 2 ] = id_2;
//        std::sort( std::begin( ids ), std::end( ids ) );    // 将三个顶点的出现顺序进行修改 - 这样后续用hash表或者其他操作的时候, 都具有唯一的顺序
//
//        Triangle_ptr triangle_ptr;
//        if ( m_triangle_hash.if_exist( ids[ 0 ], ids[ 1 ], ids[ 2 ] ) )
//        {
//            // This triangle exist
//            // triangle_ptr = m_triangle_hash.m_map_3d_hash_map[ ids[ 0 ] ][ ids[ 1 ] ][ ids[ 2 ] ];
//            triangle_ptr = *m_triangle_hash.get_data( ids[ 0 ], ids[ 1 ], ids[ 2 ] );
//        }
//        else
//        {
//            // This triangle is not exist.
//            // Step 1: new a triangle
//            triangle_ptr = std::make_shared< Triangle >( ids[ 0 ], ids[ 1 ], ids[ 2 ] );
//            triangle_ptr->m_vis_score = 1;
//            m_mutex_triangle_hash.lock();
//            m_triangle_hash.insert( ids[ 0 ], ids[ 1 ], ids[ 2 ], triangle_ptr );
//            m_mutex_triangle_hash.unlock();
//            // return m_map_pt_triangle.size();
//            // return m_triangle_list.size();
//            // return triangle_ptr;
//        }
//
//        // 这里相当于是这个triangle无论存在还是不存在都是都会进行insert | 已经存在的triangle_ptr在这里也会被添加进去(那就更新其颜色信息)
//        m_mutex_triangle_hash.lock();
//        // 给 triangle_ptr 顶点颜色信息
//        for(auto i = 0; i < 3 ;++i)
//        {
//            triangle_ptr->m_tri_pts_color[i][0] = color[i][0];
//            triangle_ptr->m_tri_pts_color[i][1] = color[i][1];
//            triangle_ptr->m_tri_pts_color[i][2] = color[i][2];
//        }
//
//
//        insert_triangle_to_list( triangle_ptr, frame_idx );
//
//        // ins
//        // 这里默认为false - 但是具体的作用应该是进行点以及边到三角形的映射关系
//        if ( build_triangle_map )
//        {
//            // Step 2: add this triangle to points list:
//            m_map_pt_triangle[ ids[ 0 ] ].insert( triangle_ptr );
//            m_map_pt_triangle[ ids[ 1 ] ].insert( triangle_ptr );
//            m_map_pt_triangle[ ids[ 2 ] ].insert( triangle_ptr );
//
//            if ( m_enable_map_edge_triangle )
//            {
//                m_map_edge_triangle.m_map_2d_hash_map[ ids[ 0 ] ][ ids[ 1 ] ].insert( triangle_ptr );
//                m_map_edge_triangle.m_map_2d_hash_map[ ids[ 0 ] ][ ids[ 2 ] ].insert( triangle_ptr );
//                m_map_edge_triangle.m_map_2d_hash_map[ ids[ 1 ] ][ ids[ 2 ] ].insert( triangle_ptr );
//                // Find conflict triangle
//                if ( m_map_edge_triangle.m_map_2d_hash_map[ ids[ 0 ] ][ ids[ 1 ] ].size() > 2 )
//                {
//                    m_map_edge_triangle_conflicted.m_map_2d_hash_map[ ids[ 0 ] ][ ids[ 1 ] ] =
//                        m_map_edge_triangle.m_map_2d_hash_map[ ids[ 0 ] ][ ids[ 1 ] ];
//                }
//                if ( m_map_edge_triangle.m_map_2d_hash_map[ ids[ 0 ] ][ ids[ 2 ] ].size() > 2 )
//                {
//                    m_map_edge_triangle_conflicted.m_map_2d_hash_map[ ids[ 0 ] ][ ids[ 2 ] ] =
//                        m_map_edge_triangle.m_map_2d_hash_map[ ids[ 0 ] ][ ids[ 2 ] ];
//                }
//                if ( m_map_edge_triangle.m_map_2d_hash_map[ ids[ 1 ] ][ ids[ 2 ] ].size() > 2 )
//                {
//                    m_map_edge_triangle_conflicted.m_map_2d_hash_map[ ids[ 1 ] ][ ids[ 2 ] ] =
//                        m_map_edge_triangle.m_map_2d_hash_map[ ids[ 1 ] ][ ids[ 2 ] ];
//                }
//            }
//        }
//        m_mutex_triangle_hash.unlock();
//
//        return triangle_ptr;
//    }

    // 函数重构
    Triangle_ptr insert_triangle( int id_0, int id_1, int id_2, int build_triangle_map = false, const int & frame_idx = 0 )
    {
        int ids[ 3 ];
        ids[ 0 ] = id_0;
        ids[ 1 ] = id_1;
        ids[ 2 ] = id_2;
        std::sort( std::begin( ids ), std::end( ids ) );    // 将三个顶点的出现顺序进行修改 - 这样后续用hash表或者其他操作的时候, 都具有唯一的顺序

        Triangle_ptr triangle_ptr;
        if ( m_triangle_hash.if_exist( ids[ 0 ], ids[ 1 ], ids[ 2 ] ) )
        {
            // This triangle exist
            // triangle_ptr = m_triangle_hash.m_map_3d_hash_map[ ids[ 0 ] ][ ids[ 1 ] ][ ids[ 2 ] ];
//            triangle_ptr = m_triangle_hash.get_data( ids[ 0 ], ids[ 1 ], ids[ 2 ] );
            triangle_ptr = *m_triangle_hash.get_data( ids[ 0 ], ids[ 1 ], ids[ 2 ] );
        }
        else
        {
            // This triangle is not exist.
            // Step 1: new a triangle
            triangle_ptr = std::make_shared< Triangle >( ids[ 0 ], ids[ 1 ], ids[ 2 ] );
            triangle_ptr->m_vis_score = 1;
            m_mutex_triangle_hash.lock();
            m_triangle_hash.insert( ids[ 0 ], ids[ 1 ], ids[ 2 ], triangle_ptr );
            m_mutex_triangle_hash.unlock();
            // return m_map_pt_triangle.size();
            // return m_triangle_list.size();
            // return triangle_ptr;
        }

        // 这里相当于是这个triangle无论存在还是不存在都是都会进行insert | 已经存在的triangle_ptr在这里也会被添加进去(但是其问题是颜色需要被更新)
        m_mutex_triangle_hash.lock();
        // m_triangle_list.insert( triangle_ptr );
        insert_triangle_to_list( triangle_ptr, frame_idx );

        // ins
        // 这里默认为false - 但是具体的作用应该是进行点以及边到三角形的映射关系
        if ( build_triangle_map )
        {
            // Step 2: add this triangle to points list:
            m_map_pt_triangle[ ids[ 0 ] ].insert( triangle_ptr );
            m_map_pt_triangle[ ids[ 1 ] ].insert( triangle_ptr );
            m_map_pt_triangle[ ids[ 2 ] ].insert( triangle_ptr );

            if ( m_enable_map_edge_triangle )
            {
                m_map_edge_triangle.m_map_2d_hash_map[ ids[ 0 ] ][ ids[ 1 ] ].insert( triangle_ptr );
                m_map_edge_triangle.m_map_2d_hash_map[ ids[ 0 ] ][ ids[ 2 ] ].insert( triangle_ptr );
                m_map_edge_triangle.m_map_2d_hash_map[ ids[ 1 ] ][ ids[ 2 ] ].insert( triangle_ptr );
                // Find conflict triangle
                if ( m_map_edge_triangle.m_map_2d_hash_map[ ids[ 0 ] ][ ids[ 1 ] ].size() > 2 )
                {
                    m_map_edge_triangle_conflicted.m_map_2d_hash_map[ ids[ 0 ] ][ ids[ 1 ] ] =
                            m_map_edge_triangle.m_map_2d_hash_map[ ids[ 0 ] ][ ids[ 1 ] ];
                }
                if ( m_map_edge_triangle.m_map_2d_hash_map[ ids[ 0 ] ][ ids[ 2 ] ].size() > 2 )
                {
                    m_map_edge_triangle_conflicted.m_map_2d_hash_map[ ids[ 0 ] ][ ids[ 2 ] ] =
                            m_map_edge_triangle.m_map_2d_hash_map[ ids[ 0 ] ][ ids[ 2 ] ];
                }
                if ( m_map_edge_triangle.m_map_2d_hash_map[ ids[ 1 ] ][ ids[ 2 ] ].size() > 2 )
                {
                    m_map_edge_triangle_conflicted.m_map_2d_hash_map[ ids[ 1 ] ][ ids[ 2 ] ] =
                            m_map_edge_triangle.m_map_2d_hash_map[ ids[ 1 ] ][ ids[ 2 ] ];
                }
            }
        }
        m_mutex_triangle_hash.unlock();

        return triangle_ptr;
    }

    std::set< int > get_conflict_triangle_pts()
    {
        std::set< int > conflict_triangle_pts;
        if ( m_enable_map_edge_triangle )
        {
            for ( auto it : m_map_edge_triangle_conflicted.m_map_2d_hash_map )
            {
                for ( auto it_it : it.second )
                {
                    Triangle_set triangle_list = it_it.second;
                    for ( auto tri : triangle_list )
                    {
                        // g_triangle_manager.erase_triangle( tri );
                        // conflict_triangle++;

                        conflict_triangle_pts.insert( tri->m_tri_pts_id[ 0 ] );
                        conflict_triangle_pts.insert( tri->m_tri_pts_id[ 1 ] );
                        conflict_triangle_pts.insert( tri->m_tri_pts_id[ 2 ] );
                    }
                    // printf_line;
                }
            }
        }
        return conflict_triangle_pts;
    }

    void clear_conflicted_triangles_list()
    {
        if ( m_enable_map_edge_triangle )
        {
            m_map_edge_triangle_conflicted.clear();
        }
    }
};
